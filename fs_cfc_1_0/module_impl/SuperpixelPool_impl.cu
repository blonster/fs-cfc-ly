#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

using namespace at;
using namespace std;

void determineBDim( int &ratio, int &rsize, int max, int a, int maxTh){
  ratio = ceil( max * a * 1.3 / (double)maxTh);
  rsize = ceil( max / (double)ratio);
}

void syncCudaAndCheckError( int pos){
  cudaError_t eid = cudaGetLastError();
  if( eid) printf( "SP(%d) Cuda error: %s\n", pos, cudaGetErrorString( eid));
  eid = cudaDeviceSynchronize();
  if( eid) printf( "SP(%d) Finally %s\n", pos, cudaGetErrorString( cudaGetLastError()));
}

template< typename T>
__global__ void SuperpixelPool_prepare_kernel_sum( T* inputsp, T* inputf, T* spsum, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, int syratio)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  int pixelcnt = realgrids * blockDim.y;
  
  T* tsum = (T*)sharedbuf;
  
  int sx = threadIdx.x;
  int gx = sx / gridSize;
  int sy_base = threadIdx.y;
  int sidx = sy_base * realgrids + sx;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;
  int ix = ( spidxx - 2) * gridSize + sx;

  int ch = blockIdx.z;
    
  T tmpsum = 0;
  for( int syd = 0; syd < syratio; syd++){
    int sy = sy_base * syratio + syd;
    int gy = sy / gridSize;
    int iy = ( spidxy - 2) * gridSize + sy;
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      T w = inputsp[ ( gy * 3 + gx) * imageHeight * imageWidth + iy * imageWidth + ix];
      tmpsum += inputf[ ( ch * imageHeight + iy) * imageWidth + ix] * w;
    }
  }
  tsum[ sidx] = tmpsum;
  
  for( int a = 1; a < pixelcnt; a *= 2){
    __syncthreads();

    int realsx = sidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < pixelcnt){
      tsum[ realsx] += tsum[ realsx_a];
    }
  }
  __syncthreads();
  if( sidx == 0){
    spsum[ ( ch * gridH + spidxy) * gridW + spidxx] += tsum[0];
  }
}

template< typename T>
__global__ void SuperpixelPool_prepare_kernel_weight( T* inputsp, T* inputf, T* spweight, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, int syratio)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  int pixelcnt = realgrids * blockDim.y;
  
  T* tweight = (T*)sharedbuf;
  
  int sx = threadIdx.x;
  int gx = sx / gridSize;
  int sy_base = threadIdx.y;
  int sidx = sy_base * realgrids + sx;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;
  int ix = ( spidxx - 2) * gridSize + sx;
    
  T tmpweight = 0;
  for( int syd = 0; syd < syratio; syd++){
    int sy = sy_base * syratio + syd;
    int gy = sy / gridSize;
    int iy = ( spidxy - 2) * gridSize + sy;    
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      T w = inputsp[ ( gy * 3 + gx) * imageHeight * imageWidth + iy * imageWidth + ix];
      tmpweight += w;
    }
  }
  tweight[ sidx] = tmpweight;
  
  for( int a = 1; a < pixelcnt; a *= 2){
    __syncthreads();

    int realsx = sidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < pixelcnt){
      tweight[ realsx] += tweight[ realsx_a];
    }
  }
  __syncthreads();
  if( sidx == 0){
    spweight[ spidxy * gridW + spidxx] += tweight[0];
  }
}

template< typename T>
__global__ void SuperpixelPool_forward_kernel( T* inputsp, T* inputf, T* output, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, T* spsum, T* spweight)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  int pixelcnt = realgrids * realgrids;
  
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;

  int ch = blockIdx.z;
    
  T w = spweight[ spidxy * gridW + spidxx];
  if( w > 0){
    T s = spsum[ ( ch * gridH + spidxy) * gridW + spidxx];
    T aver = s / w;
    output[ ( ch * gridH + spidxy) * gridW + spidxx] += aver;
  }
} 

Tensor SuperpixelPool_forward_cuda(Tensor inputsp,
				   Tensor inputf,
				   int K)
{
  inputsp = inputsp.contiguous();
  inputf = inputf.contiguous();

  int nBatch = inputsp.size(0);
  int nChannel = inputf.size(1);
  int maxH = inputsp.size(2);
  int maxW = inputsp.size(3);

  // Partition the images.
  int grids = ceil( sqrt( maxH * maxW / (double)K));
  int n[2] = { ceil( maxH / (double)grids) + 2, ceil( maxW / (double)grids) + 2};
  int realgrids = 3 * grids;
  int npgrid = realgrids * realgrids;

  int outputHeight = n[0];
  int outputWidth = n[1];

  Tensor output = zeros( { nBatch, nChannel, outputHeight, outputWidth}, inputsp.options());

  for( int ibatch = 0; ibatch < nBatch; ibatch++){

    int cols = n[1];
    int rows = n[0];
    dim3 griddim_pw( cols, rows);
    dim3 griddim_pf( cols, rows, nChannel);
    dim3 griddim( cols, rows, nChannel);
    int syratio = 1;
    int rsize_sy = 1;
    determineBDim( syratio, rsize_sy, realgrids, realgrids, 1024);
    dim3 blockdim_p( realgrids, rsize_sy);
    dim3 blockdim( 1);

    Tensor spsum = zeros( {nChannel, rows, cols}, inputsp.options());
    Tensor spweight = zeros( {rows, cols}, inputsp.options());
    
    if( inputsp.dtype() == ScalarType::Float){
      SuperpixelPool_prepare_kernel_weight<float><<< griddim_pw, blockdim_p, npgrid * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride(0), inputf.data<float>() + ibatch * inputf.stride(0), spweight.data<float>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(1);

      SuperpixelPool_prepare_kernel_sum<float><<< griddim_pf, blockdim_p, npgrid * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride(0), inputf.data<float>() + ibatch * inputf.stride(0), spsum.data<float>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(2);

      SuperpixelPool_forward_kernel<float><<< griddim, blockdim, 0>>>( inputsp.data<float>() + ibatch * inputsp.stride(0), inputf.data<float>() + ibatch * inputf.stride(0), output.data<float>() + ibatch * output.stride( 0), maxH, maxW, n[0], n[1], grids, spsum.data<float>(), spweight.data<float>());
      syncCudaAndCheckError(3);
    } else if( inputsp.dtype() == ScalarType::Double){
      SuperpixelPool_prepare_kernel_weight<double><<< griddim_pw, blockdim_p, npgrid * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride(0), inputf.data<double>() + ibatch * inputf.stride(0), spweight.data<double>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(1);

      SuperpixelPool_prepare_kernel_sum<double><<< griddim_pf, blockdim_p, npgrid * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride(0), inputf.data<double>() + ibatch * inputf.stride(0), spsum.data<double>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(2);

      SuperpixelPool_forward_kernel<double><<< griddim, blockdim, 0>>>( inputsp.data<double>() + ibatch * inputsp.stride(0), inputf.data<double>() + ibatch * inputf.stride(0), output.data<double>() + ibatch * output.stride( 0), maxH, maxW, n[0], n[1], grids, spsum.data<double>(), spweight.data<double>());
      syncCudaAndCheckError(3);
    }
  }

  return output;
}

template< typename T>
__global__ void SuperpixelPool_backward_kernel_f( T* inputsp, T* inputf, T* gradOutput, T* gradInputf, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, T* spweight)
{
  extern __shared__ char sharedbuf[];
  T* tsum = (T*)sharedbuf;
  
  int realgrids = gridSize * 3;
  
  int gx = threadIdx.x;
  int gy = threadIdx.y;
  int gidx = gy * 3 + gx;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int ch = blockIdx.z;
  int spidxx = ix / gridSize + 2 - gx;
  int spidxy = iy / gridSize + 2 - gy;

  tsum[ gidx] = 0;
  if( spidxx >= 0 && spidxx < gridW && spidxy >= 0 && spidxy < gridH){
    T sw = spweight[ spidxy * gridW + spidxx];
    if( sw > 0){
      T gout = gradOutput[ ( ch * gridH + spidxy) * gridW + spidxx];
      T w = inputsp[ ( gidx * imageHeight + iy) * imageWidth + ix];
      tsum[ gidx] += gout * w / sw;
    }
  }
  for( int a = 1; a < 9; a *= 2){
    __syncthreads();

    int realsx = gidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < 9){
      tsum[ realsx] += tsum[ realsx_a];
    }
  }
  __syncthreads();

  if( gidx == 0){
      int fidx = ( ch * imageHeight + iy) * imageWidth + ix;
      gradInputf[ fidx] += tsum[0];      
  }
}

template< typename T>
__global__ void SuperpixelPool_backward_kernel_sp( T* inputsp, T* inputf, int channels, T* gradOutput, T* gradInputsp, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, T* spsum, T* spweight)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  int pixelcnt = realgrids * realgrids;
  
  T* tsum = (T*)sharedbuf;
  
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int gidx = blockIdx.z;
  int gy = gidx / 3;
  int gx = gidx - gy * 3;
  int spidxx = ix / gridSize + 2 - gx;
  int spidxy = iy / gridSize + 2 - gy;

  int ch = threadIdx.x;
    
  T tmpsum = 0;
  if( spidxx >= 0 && spidxx < gridW && spidxy >= 0 && spidxy < gridH){
    T sw = spweight[ spidxy * gridW + spidxx];
    if( sw > 0){
      T gout = gradOutput[ ( ch * gridH + spidxy) * gridW + spidxx];
      T ss = spsum[ ( ch * gridH + spidxy) * gridW + spidxx];
      T f = inputf[ ( ch * imageHeight + iy) * imageWidth + ix];
      tmpsum += gout * ( f / sw - ss / ( sw * sw));
    }
  }
  tsum[ ch] = tmpsum;
  
  for( int a = 1; a < channels; a *= 2){
    __syncthreads();

    int realsx = ch * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < channels){
      tsum[ realsx] += tsum[ realsx_a];
    }
  }
  __syncthreads();

  if( ch == 0){
    int spidx = ( gidx * imageHeight + iy) * imageWidth + ix;
    gradInputsp[ spidx] += tsum[0];
  }
}

vector<Tensor> SuperpixelPool_backward_cuda( Tensor inputsp,
					     Tensor inputf,
					     Tensor gradOutput,
					     int K)
{
  gradOutput = gradOutput.contiguous();
  inputsp = inputsp.contiguous();
  inputf = inputf.contiguous();

  int nBatch = inputsp.size(0);
  int nChannel = inputf.size(1);
  int maxH = inputsp.size(2);
  int maxW = inputsp.size(3);
  int nInputspPlane = inputsp.size(1);

  // Partition the images.
  int grids = ceil( sqrt( maxH * maxW / K));
  int n[2] = { gradOutput.size(2), gradOutput.size(3)};
  int realgrids = 3 * grids;
  int npgrid = realgrids * realgrids;

  int outputHeight = n[0];
  int outputWidth = n[1];

  Tensor gradInputsp = zeros( inputsp.sizes(), inputsp.options());
  Tensor gradInputf = zeros( inputf.sizes(), inputf.options());

  for( int ibatch = 0; ibatch < nBatch; ibatch++){
    int cols = n[1];
    int rows = n[0];
    dim3 griddim_pw( cols, rows);
    dim3 griddim_pf( cols, rows, nChannel);
    dim3 griddim_f( maxW, maxH, nChannel);
    dim3 griddim_sp( maxW, maxH, 9);
    int syratio = 1;
    int rsize_sy = 1;
    determineBDim( syratio, rsize_sy, realgrids, realgrids, 1024);
    dim3 blockdim_p( realgrids, rsize_sy);
    dim3 blockdim_f( 3, 3);
    dim3 blockdim_sp( nChannel);
    Tensor spsum = zeros( {nChannel, rows, cols}, inputsp.options());
    Tensor spweight = zeros( {rows, cols}, inputsp.options());
    if( inputsp.dtype() == ScalarType::Float){
      SuperpixelPool_prepare_kernel_weight<float><<< griddim_pw, blockdim_p, npgrid * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride(0), inputf.data<float>() + ibatch * inputf.stride(0), spweight.data<float>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(4);

      SuperpixelPool_prepare_kernel_sum<float><<< griddim_pf, blockdim_p, npgrid * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride(0), inputf.data<float>() + ibatch * inputf.stride(0), spsum.data<float>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(5);

      SuperpixelPool_backward_kernel_f<float><<< griddim_f, blockdim_f, 9 * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputf.data<float>() + ibatch * gradInputf.stride( 0), maxH, maxW, n[0], n[1], grids, spweight.data<float>());
      syncCudaAndCheckError(6);

      SuperpixelPool_backward_kernel_sp<float><<< griddim_sp, blockdim_sp, nChannel * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputsp.data<float>() + ibatch * gradInputsp.stride( 0), maxH, maxW, n[0], n[1], grids, spsum.data<float>(), spweight.data<float>());
      syncCudaAndCheckError(7);
    } else if( inputsp.dtype() == ScalarType::Double){
      SuperpixelPool_prepare_kernel_weight<double><<< griddim_pw, blockdim_p, npgrid * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride(0), inputf.data<double>() + ibatch * inputf.stride(0), spweight.data<double>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(4);

      SuperpixelPool_prepare_kernel_sum<double><<< griddim_pf, blockdim_p, npgrid * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride(0), inputf.data<double>() + ibatch * inputf.stride(0), spsum.data<double>(), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(5);

      SuperpixelPool_backward_kernel_f<double><<< griddim_f, blockdim_f, 9 * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputf.data<double>() + ibatch * gradInputf.stride( 0), maxH, maxW, n[0], n[1], grids, spweight.data<double>());
      syncCudaAndCheckError(6);

      SuperpixelPool_backward_kernel_sp<double><<< griddim_sp, blockdim_sp, nChannel * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputsp.data<double>() + ibatch * gradInputsp.stride( 0), maxH, maxW, n[0], n[1], grids, spsum.data<double>(), spweight.data<double>());
      syncCudaAndCheckError(7);
    }
    cudaDeviceSynchronize();
  }

  return { gradInputsp, gradInputf};
}
