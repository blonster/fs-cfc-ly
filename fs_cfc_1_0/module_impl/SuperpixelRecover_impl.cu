#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

using namespace std;
using namespace at;

void determineBDim( int &ratio, int &rsize, int max, int a, int maxTh){
  ratio = ceil( max * a * 1.3 / (double)maxTh);
  rsize = ceil( max / (double)ratio);
}

void syncCudaAndCheckError( int pos){
  cudaError_t eid = cudaGetLastError();
  if( eid) printf( "SR(%d) Cuda error: %s\n", pos, cudaGetErrorString( eid));
  eid = cudaDeviceSynchronize();
  if( eid) printf( "SR(%d) Finally %s\n", pos, cudaGetErrorString( cudaGetLastError()));
}

template< typename T>
__global__ void SuperpixelRecover_forward_kernel( T* inputsp, T* inputf, int channel, T* output, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  
  T* tmpsum = (T*)sharedbuf;
  
  int gx = threadIdx.x;
  int gy = threadIdx.y;
  int gidx = gy * 3 + gx;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int ch = blockIdx.z;
  int spidxx = ix / gridSize + 2 - gx;
  int spidxy = iy / gridSize + 2 - gy;

  tmpsum[ gidx] = 0;
  if( spidxx >= 0 && spidxx < gridW && spidxy >= 0 && spidxy < gridH){
    T f = inputf[ ( ch * gridH + spidxy) * gridW + spidxx];
    T sp = inputsp[ ( ( gy * 3 + gx) * imageHeight + iy) * imageWidth + ix];
    tmpsum[ gidx] += f * sp;
  }

  for( int a = 1; a < 9; a *= 2){
    __syncthreads();

    int realsx = gidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < 9){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }
  __syncthreads();
  if( gidx == 0){
    output[ ( ch * imageHeight + iy) * imageWidth + ix] += tmpsum[0];
  }
  
}

Tensor SuperpixelRecover_forward_cuda( Tensor inputsp,
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
  int n[2] = { inputf.size(2), inputf.size(3)};
  int realgrids = 3 * grids;
  int npgrid = realgrids * realgrids;

  Tensor output = zeros( { nBatch, nChannel, maxH, maxW}, inputsp.options());

  for( int ibatch = 0; ibatch < nBatch; ibatch++){
    int cols = maxW;
    int rows = maxH;
    dim3 griddim( cols, rows, nChannel);
    dim3 blockdim( 3, 3);
    if( inputsp.dtype() == ScalarType::Float){
      SuperpixelRecover_forward_kernel<float><<< griddim, blockdim, 9 * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), nChannel, output.data<float>() + ibatch * output.stride( 0), maxH, maxW, n[0], n[1], grids);
      syncCudaAndCheckError(1);

    } else if( inputsp.dtype() == ScalarType::Double){
      SuperpixelRecover_forward_kernel<double><<< griddim, blockdim, 9 * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), nChannel, output.data<double>() + ibatch * output.stride( 0), maxH, maxW, n[0], n[1], grids);
      syncCudaAndCheckError(1);

    }
  }
  return output;
}

template< typename T>
__global__ void SuperpixelRecover_backward_kernel_f( T* inputsp, T* inputf, int channel, T* gradOutput, T* gradInputf, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize, int syratio)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  int pixelcnt = realgrids * blockDim.y;

  T* tmpsum = (T*)sharedbuf;

  int sx = threadIdx.x;
  int gx = sx / gridSize;
  int sy_base = threadIdx.y;
  int sidx = sy_base * realgrids + sx;
  int spidxx = blockIdx.x;
  int spidxy = blockIdx.y;
  int ix = ( spidxx - 2) * gridSize + sx;

  int ch = blockIdx.z;

  tmpsum[ sidx] = 0;
  for( int syd = 0; syd < syratio; syd++){
    int sy = sy_base * syratio + syd;
    int gy = sy / gridSize;
    int iy = ( spidxy - 2) * gridSize + sy;
    if( ix >= 0 && ix < imageWidth && iy >= 0 && iy < imageHeight){
      T gout = gradOutput[ ( ch * imageHeight + iy) * imageWidth + ix];
      int inspidx = ( ( gy * 3 + gx) * imageHeight + iy) * imageWidth + ix;
      T sp = inputsp[ inspidx];

      tmpsum[ sidx] += sp * gout;
    }
  }
  for( int a = 1; a < pixelcnt; a *= 2){
    __syncthreads();

    int realsx = sidx * a * 2;
    int realsx_a = realsx + a;
    if( realsx_a < pixelcnt){
      tmpsum[ realsx] += tmpsum[ realsx_a];
    }
  }
  __syncthreads();
  if( sidx == 0){
    gradInputf[ ( ch * gridH + spidxy) * gridW + spidxx] += tmpsum[ 0];
  }
}

template< typename T>
__global__ void SuperpixelRecover_backward_kernel_sp( T* inputsp, T* inputf, int channel, T* gradOutput, T* gradInputsp, int imageHeight, int imageWidth, int gridH, int gridW, int gridSize)
{
  extern __shared__ char sharedbuf[];
  
  int realgrids = gridSize * 3;
  
  int gx = threadIdx.x;
  int gy = threadIdx.y;
  int gidx = gy * 3 + gx;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int ch = blockIdx.z;
  int spidxx = ix / gridSize + 2 - gx;
  int spidxy = iy / gridSize + 2 - gy;

  int inspidx = ( ( gy * 3 + gx) * imageHeight + iy) * imageWidth + ix;
  if( spidxx >= 0 && spidxx < gridW && spidxy >= 0 && spidxy < gridH){
    T gout = gradOutput[ ( ch * imageHeight + iy) * imageWidth + ix];
    T f = inputf[ ( ch * gridH + spidxy) * gridW + spidxx];
    gradInputsp[ inspidx] += f * gout;    
  }  
}

vector<Tensor> SuperpixelRecover_backward_cuda( Tensor inputsp,
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

  // Partition the images.
  int grids = ceil( sqrt( maxH * maxW / (double)K));
  int n[2] = { inputf.size(2), inputf.size(3)};
  int realgrids = 3 * grids;
  int npgrid = realgrids * realgrids;

  Tensor gradInputsp = zeros( inputsp.sizes(), inputsp.options());
  Tensor gradInputf = zeros( inputf.sizes(), inputf.options());

  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  double maxTh = properties.maxThreadsPerMultiProcessor;
  for( int ibatch = 0; ibatch < nBatch; ibatch++){
    int cols = n[1];
    int rows = n[0];
    dim3 griddim_f( cols, rows, nChannel);
    dim3 griddim_sp( maxW, maxH, nChannel);
    int syratio = 1;
    int rsize_sy = 1;
    determineBDim( syratio, rsize_sy, realgrids, realgrids, 1024);
    dim3 blockdim_f( realgrids, rsize_sy);
    dim3 blockdim_sp( 3, 3);
    if( inputsp.dtype() == ScalarType::Float){
      SuperpixelRecover_backward_kernel_f<float><<< griddim_f, blockdim_f, npgrid * sizeof( float)>>>( inputsp.data<float>() + ibatch * inputsp.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputf.data<float>() + ibatch * gradInputf.stride( 0), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(2);

      SuperpixelRecover_backward_kernel_sp<float><<< griddim_sp, blockdim_sp, 0>>>( inputsp.data<float>() + ibatch * inputsp.stride( 0), inputf.data<float>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<float>() + ibatch * gradOutput.stride( 0), gradInputsp.data<float>() + ibatch * gradInputsp.stride( 0), maxH, maxW, n[0], n[1], grids);
      syncCudaAndCheckError(3);

    } else if( inputsp.dtype() == ScalarType::Double){
      SuperpixelRecover_backward_kernel_f<double><<< griddim_f, blockdim_f, npgrid * sizeof( double)>>>( inputsp.data<double>() + ibatch * inputsp.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputf.data<double>() + ibatch * gradInputf.stride( 0), maxH, maxW, n[0], n[1], grids, syratio);
      syncCudaAndCheckError(2);

      SuperpixelRecover_backward_kernel_sp<double><<< griddim_sp, blockdim_sp, 0>>>( inputsp.data<double>() + ibatch * inputsp.stride( 0), inputf.data<double>() + ibatch * inputf.stride( 0), nChannel, gradOutput.data<double>() + ibatch * gradOutput.stride( 0), gradInputsp.data<double>() + ibatch * gradInputsp.stride( 0), maxH, maxW, n[0], n[1], grids);
      syncCudaAndCheckError(3);

    }
    cudaDeviceSynchronize();
  }

  return { gradInputsp, gradInputf};
}


