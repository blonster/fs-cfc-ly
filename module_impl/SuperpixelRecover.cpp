#include <torch/torch.h>

#include <vector>

using namespace std;
using namespace at;

Tensor SuperpixelRecover_forward_cuda( Tensor inputsp,
				       Tensor inputf,
				       int K);
vector<Tensor> SuperpixelRecover_backward_cuda( Tensor inputsp,
						Tensor inputf,
						Tensor gradOutput,
						int K);

Tensor SuperpixelRecover_forward( Tensor inputsp,
				  Tensor inputf,
				  int K)
{
  return SuperpixelRecover_forward_cuda( inputsp, inputf, K);
}

vector<Tensor> SuperpixelRecover_backward( Tensor inputsp,
				   Tensor inputf,
				   Tensor gradOutput,
				   int K)
{
  return SuperpixelRecover_backward_cuda( inputsp, inputf, gradOutput, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SuperpixelRecover_forward, "SuperpixelRecover forward");
  m.def("backward", &SuperpixelRecover_backward, "SuperpixelRecover backward");
}


