#include <torch/torch.h>

#include <vector>

using namespace std;
using namespace at;

Tensor ControlledFilterConv_forward_cuda( Tensor inputw,
					  Tensor inputf,
					  Tensor paramw,
					  int nplane_o,
					  int kx, int ky);

vector<Tensor> ControlledFilterConv_backward_cuda( Tensor inputw,
						   Tensor inputf,
						   Tensor paramw,
						   Tensor gradOutput,
						   int kx, int ky);

Tensor ControlledFilterConv_forward( Tensor inputw,
				     Tensor inputf,
				     Tensor paramw,
				     int nplane_o,
				     int kx, int ky)
{
  return ControlledFilterConv_forward_cuda( inputw, inputf, paramw, nplane_o, kx, ky);
}

vector<Tensor> ControlledFilterConv_backward( Tensor inputw,
					      Tensor inputf,
					      Tensor paramw,
					      Tensor gradOutput,
					      int kx, int ky)
{
  return ControlledFilterConv_backward_cuda( inputw, inputf, paramw, gradOutput, kx, ky);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ControlledFilterConv_forward, "ControlledFilterConv forward");
  m.def("backward", &ControlledFilterConv_backward, "ControlledFilterConv backward");
}


