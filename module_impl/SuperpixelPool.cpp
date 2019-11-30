#include <torch/torch.h>
#include <vector>

using namespace std;
using namespace at;

Tensor SuperpixelPool_forward_cuda(Tensor inputsp,
				   Tensor inputf,
				   int K);
vector<Tensor> SuperpixelPool_backward_cuda( Tensor inputsp,
					     Tensor inputf,
					     Tensor gradOutput,
					     int K);


Tensor SuperpixelPool_forward( Tensor inputsp,
			       Tensor inputf,
			       int K)
{  
  return SuperpixelPool_forward_cuda( inputsp, inputf, K);
}

vector<Tensor> SuperpixelPool_backward( Tensor inputsp,
					Tensor inputf,
					Tensor gradOutput,
					int K)
{
  return SuperpixelPool_backward_cuda( inputsp, inputf, gradOutput, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SuperpixelPool_forward, "SuperpixelPool forward");
  m.def("backward", &SuperpixelPool_backward, "SuperpixelPool backward");
}

