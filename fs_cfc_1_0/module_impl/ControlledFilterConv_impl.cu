#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

using namespace std;
using namespace at;

void determineBDim( int &ratio_x, int &ratio_y, int &ratio_z, dim3 &blockdim, int maxTh){
  ratio_x = 1;
  ratio_y = 1;
  ratio_z = 1;
  if( blockdim.x * blockdim.y * blockdim.z <= maxTh){
    return;
  } else{
    int other = blockdim.x * blockdim.y;
    if( other > maxTh / 2){
      ratio_z = blockdim.z;
      blockdim.z = 1;
    } else{
      int rsize = blockdim.z;
      int maxs = blockdim.z;
      for( ratio_z = 1; ratio_z < maxs && ( other * rsize  > maxTh); ){
	ratio_z++;
	rsize = ceil( maxs / (double)ratio_z);
      }
      blockdim.z = rsize;
    }

    other = blockdim.x * blockdim.z;
    if( other > maxTh / 2){
      ratio_y = blockdim.y;
      blockdim.y = 1;
    } else{
      int rsize = blockdim.y;
      int maxs = blockdim.y;
      for( ratio_y = 1; ratio_y < maxs && ( other * rsize  > maxTh); ){
	ratio_y++;
	rsize = ceil( maxs / (double)ratio_y);
      }
      blockdim.y = rsize;
    }

    other = blockdim.y * blockdim.z;
    if( other > maxTh / 2){
      ratio_x = blockdim.x;
      blockdim.x = 1;
    } else{
      int rsize = blockdim.x;
      int maxs = blockdim.x;
      for( ratio_x = 1; ratio_x < maxs && ( other * rsize  > maxTh); ){
	ratio_x++;
	rsize = ceil( maxs / (double)ratio_x);
      }
      blockdim.x = rsize;
    }

  }
}

void syncCudaAndCheckError( int pos){
  cudaError_t eid = cudaGetLastError();
  if( eid) printf( "CFC(%d) Cuda error: %s\n", pos, cudaGetErrorString( eid));
  eid = cudaDeviceSynchronize();
  if( eid) printf( "CFC(%d) Finally %s\n", pos, cudaGetErrorString( cudaGetLastError()));
}

template< typename T>
__global__ void ControlledFilterConv_forward_kernel( int ky, int kx, T* inputw, T* inputf, T* paramw, T* output, int iChannel, int oChannel, int wChannel, int imageHeight, int imageWidth, int wratio, int iratio, int kratio)
{
  int ks = kx * ky;
  int riChannel = iChannel + 1;
  int rwChannel = wChannel + 1;
  int cross = blockDim.y * blockDim.x;
  int total = cross * blockDim.z;

  extern __shared__ char sharedbuf[];
    
  T* tmpsum = (T*)sharedbuf;
  
  int wch_base = threadIdx.x;
  int ich_base = threadIdx.y;
  int sidx_base = threadIdx.z;
  int bidx = sidx_base * cross + ich_base * blockDim.x + wch_base;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;

  int ch = blockIdx.z;
    
  tmpsum[ bidx] = 0;
  for( int sidxd = 0; sidxd < kratio; sidxd++){
    int sidx = sidx_base * kratio + sidxd;
    if( sidx >= ks) break;
    
    int sy = sidx / kx;
    int sx = sidx - sy * kx;
    int gx = sx - ( kx - 1) / 2;
    int gy = sy - ( ky - 1) / 2;
    int ix = spidxx + gx;
    int iy = spidxy + gy;
    
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      for( int ichd = 0; ichd < iratio; ichd++){
	int ich = ich_base * iratio + ichd;
	if( ich > iChannel) break;
	for( int wchd = 0; wchd < wratio; wchd++){
	  int wch = wch_base * wratio + wchd;
	  if( wch > wChannel) break;
	  if( wch < wChannel && ich < iChannel){
	    T inw = inputw[ ( wch * imageHeight + iy) * imageWidth + ix];
	    T inf = inputf[ ( ich * imageHeight + iy) * imageWidth + ix];
	    T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	    tmpsum[ bidx] += inw * inf * pmw;
	  } else if( wch == wChannel && ich == iChannel){
	    T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	    tmpsum[ bidx] += pmw;
	  } else if( wch == wChannel){
	    T inf = inputf[ ( ich * imageHeight + iy) * imageWidth + ix];
	    T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	    tmpsum[ bidx] += inf * pmw;
	  } else if( ich == iChannel){
	    T inw = inputw[ ( wch * imageHeight + iy) * imageWidth + ix];
	    T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	    tmpsum[ bidx] += inw * pmw;
	  }
	}
      }
    }
  }
  
  for( int a = 1; a < total; a *= 2){
    __syncthreads();

    int realsx = bidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < total){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }
  __syncthreads();
  if( bidx == 0){
    output[ ( ch * imageHeight + spidxy) * imageWidth + spidxx] += tmpsum[0];
  }
} 

Tensor ControlledFilterConv_forward_cuda( Tensor inputw,
					  Tensor inputf,
					  Tensor paramw,
					  int nplane_o,
					  int kx, int ky)
{
  int ks = kx * ky;

  inputw = inputw.contiguous();
  inputf = inputf.contiguous();
  paramw = paramw.contiguous();

  int nBatch = inputw.size(0);
  int wChannel = inputw.size(1);
  int iChannel = inputf.size(1);
  int oChannel = nplane_o;
  int maxH = inputw.size(2);
  int maxW = inputw.size(3);

  int n[2] = { maxH, maxW};

  Tensor output = zeros( { nBatch, oChannel, maxH, maxW}, inputw.options());

  for( int ibatch = 0; ibatch < nBatch; ibatch++){
    int cols = n[1];
    int rows = n[0];
    dim3 griddim( cols, rows, oChannel);
    dim3 blockdim( wChannel + 1, iChannel + 1, ks);
    int xratio = 1;
    int yratio = 1;
    int zratio = 1;
    determineBDim( xratio, yratio, zratio, blockdim, 512);
    if( inputw.dtype() == ScalarType::Float){
      ControlledFilterConv_forward_kernel<float><<< griddim, blockdim, blockdim.x * blockdim.y * blockdim.z * sizeof( float)>>>( ky, kx, inputw.data<float>() + ibatch * inputw.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), paramw.data<float>(), output.data<float>() + ibatch * output.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, xratio, yratio, zratio);
      syncCudaAndCheckError(1);
    } else if( inputw.dtype() == ScalarType::Double){
      ControlledFilterConv_forward_kernel<double><<< griddim, blockdim, blockdim.x * blockdim.y * blockdim.z * sizeof( double)>>>( ky, kx, inputw.data<double>() + ibatch * inputw.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), paramw.data<double>(), output.data<double>() + ibatch * output.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, xratio, yratio, zratio);
      syncCudaAndCheckError(1);
    }
  }

  return output;
}

template< typename T>
__global__ void ControlledFilterConv_backward_kernel_w( int ky, int kx, T* inputw, T* inputf, T* paramw, T* gradOutput, T* gradInputw, int iChannel, int oChannel, int wChannel, int imageHeight, int imageWidth, int oratio, int iratio, int kratio)
{
  int ks = kx * ky;
  int riChannel = iChannel + 1;
  int rwChannel = wChannel + 1;
  int cross = blockDim.y * blockDim.z;
  int total = cross * blockDim.x;

  extern __shared__ char sharedbuf[];
    
  T* tmpsum = (T*)sharedbuf;
  
  int sidx_base = threadIdx.x;
  int ich_base = threadIdx.y;
  int ch_base = threadIdx.z;
  int bidx = sidx_base * cross + ich_base * blockDim.z + ch_base;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;

  int wch = blockIdx.z;
        
  tmpsum[ bidx] = 0;
  for( int sidxd = 0; sidxd < kratio; sidxd++){
    int sidx = sidx_base * kratio + sidxd;
    if( sidx >= ks) break;
    int sy = sidx / kx;
    int sx = sidx - sy * kx;
    sy = ky - 1 - sy;
    sx = kx - 1 - sx;
    int gx = sx - ( kx - 1) / 2;
    int gy = sy - ( ky - 1) / 2;
    int ix = spidxx + gx;
    int iy = spidxy + gy;
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      for( int chd = 0; chd < oratio; chd++){
	int ch = ch_base * oratio + chd;
	if( ch >= oChannel) break;
	T gout = gradOutput[ ( ch * imageHeight + iy) * imageWidth + ix];
	for( int ichd = 0; ichd < iratio; ichd++){
	  int ich = ich_base * iratio + ichd;
	  if( ich > iChannel) break;
	
	  T w = 0;
	  T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	  if( ich < iChannel){
	    T inf = inputf[ ( ich * imageHeight + spidxy) * imageWidth + spidxx];
	    w += inf * pmw;
	  } else{
	    w += pmw;
	  }
	  tmpsum[ bidx] += w * gout;
	}
      }
    }
  }
  
  for( int a = 1; a < total; a *= 2){
    __syncthreads();

    int realsx = bidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < total){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }
  __syncthreads();
  if( bidx == 0){
    gradInputw[ ( wch * imageHeight + spidxy) * imageWidth + spidxx] += tmpsum[0];
  }
}

template< typename T>
__global__ void ControlledFilterConv_backward_kernel_f( int ky, int kx, T* inputw, T* inputf, T* paramw, T* gradOutput, T* gradInputf, int iChannel, int oChannel, int wChannel, int imageHeight, int imageWidth, int oratio, int wratio, int kratio)
{
  int ks = kx * ky;
  int riChannel = iChannel + 1;
  int rwChannel = wChannel + 1;
  int cross = blockDim.y * blockDim.z;
  int total = cross * blockDim.x;

  extern __shared__ char sharedbuf[];
    
  T* tmpsum = (T*)sharedbuf;
  
  int sidx_base = threadIdx.x;
  int wch_base = threadIdx.y;
  int ch_base = threadIdx.z;
  int bidx = sidx_base * cross + wch_base * blockDim.z + ch_base;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;

  int ich = blockIdx.z;
        
  tmpsum[ bidx] = 0;
  for( int sidxd = 0; sidxd < kratio; sidxd++){
    int sidx = sidx_base * kratio + sidxd;
    if( sidx >= ks) break;
    int sy = sidx / kx;
    int sx = sidx - sy * kx;
    sy = ky - 1 - sy;
    sx = kx - 1 - sx;
    int gx = sx - ( kx - 1) / 2;
    int gy = sy - ( ky - 1) / 2;
    int ix = spidxx + gx;
    int iy = spidxy + gy;
    
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      for( int chd = 0; chd < oratio; chd++){
	int ch = ch_base * oratio + chd;
	if( ch >= oChannel) break;
	T gout = gradOutput[ ( ch * imageHeight + iy) * imageWidth + ix];
	for( int wchd = 0; wchd < wratio; wchd++){
	  int wch = wch_base * wratio + wchd;
	  if( wch > wChannel) break;
	
	  T w = 0;
	  T pmw = paramw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch];
	  if( wch < wChannel){
	    T inw = inputw[ ( wch * imageHeight + spidxy) * imageWidth + spidxx];
	    w += inw * pmw;
	  } else{
	    w += pmw;
	  }
	  tmpsum[ bidx] += w * gout;
	}
      }
    }
  }
  
  for( int a = 1; a < total; a *= 2){
    __syncthreads();

    int realsx = bidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < total){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }

  __syncthreads();
  if( bidx == 0){
    gradInputf[ ( ich * imageHeight + spidxy) * imageWidth + spidxx] += tmpsum[0];
  }
}

template< typename T>
__global__ void ControlledFilterConv_backward_kernel_p( int ky, int kx, T* inputw, T* inputf, T* paramw, T* gradOutput, T* gradParamw, int iChannel, int oChannel, int wChannel, int imageHeight, int imageWidth, int yratio, int xratio)
{
  int ks = kx * ky;
  int riChannel = iChannel + 1;
  int rwChannel = wChannel + 1;
  int pixelcnt = blockDim.y * blockDim.x;

  extern __shared__ char sharedbuf[];
  T* tmpsum = (T*)sharedbuf;
    
  int x_base = threadIdx.x;
  int y_base = threadIdx.y;
  int bidx_offset = y_base * blockDim.x + x_base;
  int bidx = bidx_offset;
  int wch = blockIdx.x;
  int ich = blockIdx.y;
  int ch_sidx = blockIdx.z;
  int ch = ch_sidx / ks;
  int sidx = ch_sidx - ch * ks;
  int sy = sidx / kx;
  int sx = sidx - sy * kx;
  int gx = sx - ( kx - 1) / 2;
  int gy = sy - ( ky - 1) / 2;

  tmpsum[ bidx] = 0;
  for( int xd = 0; xd < xratio; xd++){
    int spidxx = x_base * xratio + xd;
    if( spidxx >= imageWidth) break;
    int ix = spidxx + gx;
    for( int yd = 0; yd < yratio; yd++){
      int spidxy = y_base * yratio + yd;
      if( spidxy >= imageHeight) break;
      int iy = spidxy + gy;
      T gout = gradOutput[ ( ch * imageHeight + spidxy) * imageWidth + spidxx];
      if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
	if( wch < wChannel && ich < iChannel){
	  T inw = inputw[ ( wch * imageHeight + iy) * imageWidth + ix];
	  T inf = inputf[ ( ich * imageHeight + iy) * imageWidth + ix];
	  tmpsum[ bidx] += inw * inf * gout;
	} else if( wch == wChannel && ich == iChannel){
	  tmpsum[ bidx] += gout;
	} else if( wch == wChannel){
	  T inf = inputf[ ( ich * imageHeight + iy) * imageWidth + ix];
	  tmpsum[ bidx] += inf * gout;
	} else if( ich == iChannel){
	  T inw = inputw[ ( wch * imageHeight + iy) * imageWidth + ix];
	  tmpsum[ bidx] += inw * gout;
	}
      }
    }
  }
  
  for( int a = 1; a < pixelcnt; a *= 2){
    __syncthreads();

    int realsx = bidx_offset * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < pixelcnt){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }

  __syncthreads();
  if( bidx_offset == 0){
    gradParamw[ ( ( sidx * oChannel + ch) * riChannel + ich) * rwChannel + wch] += tmpsum[0];
  }
}

vector<Tensor> ControlledFilterConv_backward_cuda( Tensor inputw,
						   Tensor inputf,
						   Tensor paramw,
						   Tensor gradOutput,
						   int kx, int ky)
{
  int ks = kx * ky;

  int nBatch = inputw.size(0);
  int wChannel = inputw.size(1);
  int iChannel = inputf.size(1);
  int oChannel = gradOutput.size(1);
  int maxH = inputw.size(2);
  int maxW = inputw.size(3);

  inputw = inputw.contiguous();
  inputf = inputf.contiguous();
  paramw = paramw.contiguous();
  gradOutput = gradOutput.contiguous();

  int n[2] = { maxH, maxW};

  Tensor gradInputw = zeros( inputw.sizes(), inputw.options());
  Tensor gradInputf = zeros( inputf.sizes(), inputf.options());
  Tensor gradParamw = zeros( paramw.sizes(), paramw.options());

  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  double maxTh = properties.maxThreadsPerMultiProcessor;
  for( int ibatch = 0; ibatch < nBatch; ibatch++){
    int cols = n[1];
    int rows = n[0];
    dim3 griddim_w( cols, rows, wChannel);
    dim3 griddim_f( cols, rows, iChannel);
    dim3 griddim_p( wChannel + 1, iChannel + 1, oChannel * ks);
    int xratio_w = 1;
    int yratio_w = 1;
    int zratio_w = 1;
    dim3 blockdim_w( ks, iChannel + 1, oChannel);
    int xratio_f = 1;
    int yratio_f = 1;
    int zratio_f = 1;
    dim3 blockdim_f( ks, wChannel + 1, oChannel);
    int xratio_p = 1;
    int yratio_p = 1;
    int zratio_p = 1;
    dim3 blockdim_p( cols, rows, 1);
    determineBDim( xratio_w, yratio_w, zratio_w, blockdim_w, 512);
    determineBDim( xratio_f, yratio_f, zratio_f, blockdim_f, 512);
    determineBDim( xratio_p, yratio_p, zratio_p, blockdim_p, 512);
 
    if( inputw.dtype() == ScalarType::Float){
      ControlledFilterConv_backward_kernel_w<float><<< griddim_w, blockdim_w, blockdim_w.x * blockdim_w.y * blockdim_w.z * sizeof( float)>>>( ky, kx, inputw.data<float>() + ibatch * inputw.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), paramw.data<float>(), gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputw.data<float>() + ibatch * gradInputw.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, zratio_w, yratio_w, xratio_w);
      syncCudaAndCheckError(2);

      ControlledFilterConv_backward_kernel_f<float><<< griddim_f, blockdim_f, blockdim_f.x * blockdim_f.y * blockdim_f.z * sizeof( float)>>>( ky, kx, inputw.data<float>() + ibatch * inputw.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), paramw.data<float>(), gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputf.data<float>() + ibatch * gradInputf.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, zratio_f, yratio_f, xratio_f);
      syncCudaAndCheckError(3);

      ControlledFilterConv_backward_kernel_p<float><<< griddim_p, blockdim_p, blockdim_p.x * blockdim_p.y * sizeof( float)>>>( ky, kx, inputw.data<float>() + ibatch * inputw.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), paramw.data<float>(), gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradParamw.data<float>(), iChannel, oChannel, wChannel, maxH, maxW, yratio_p, xratio_p);
      syncCudaAndCheckError(4);

    } else if( inputw.dtype() == ScalarType::Double){
      ControlledFilterConv_backward_kernel_w<double><<< griddim_w, blockdim_w, blockdim_w.x * blockdim_w.y * blockdim_w.z * sizeof( double)>>>( ky, kx, inputw.data<double>() + ibatch * inputw.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), paramw.data<double>(), gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputw.data<double>() + ibatch * gradInputw.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, zratio_w, yratio_w, xratio_w);
      syncCudaAndCheckError(2);

      ControlledFilterConv_backward_kernel_f<double><<< griddim_f, blockdim_f, blockdim_f.x * blockdim_f.y * blockdim_f.z * sizeof( double)>>>( ky, kx, inputw.data<double>() + ibatch * inputw.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), paramw.data<double>(), gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputf.data<double>() + ibatch * gradInputf.stride( 0), iChannel, oChannel, wChannel, maxH, maxW, zratio_f, yratio_f, xratio_f);
      syncCudaAndCheckError(3);

      ControlledFilterConv_backward_kernel_p<double><<< griddim_p, blockdim_p, blockdim_p.x * blockdim_p.y * sizeof( double)>>>( ky, kx, inputw.data<double>() + ibatch * inputw.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), paramw.data<double>(), gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradParamw.data<double>(), iChannel, oChannel, wChannel, maxH, maxW, yratio_p, xratio_p);
      syncCudaAndCheckError(4);

    }
    cudaDeviceSynchronize();
  }

  return { gradInputw, gradInputf, gradParamw};
}
