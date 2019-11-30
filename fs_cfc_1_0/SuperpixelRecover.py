import torch
import torch.nn as nn
from torch.autograd import Function

import srl_impl

class SuperpixelRecoverFunction( Function):
    @staticmethod
    def forward( ctx, inputsp, inputf, K):
        ctx.save_for_backward( inputsp, inputf)
        ctx.K = K
        return srl_impl.forward( inputsp, inputf, K)

    def backward( ctx, grad_output):
        inputsp, inputf = ctx.saved_tensors
        K = ctx.K
        gradinputsp, gradinputf = srl_impl.backward( inputsp, inputf, grad_output, K)
        return gradinputsp, gradinputf, None

SuperpixelRecover = SuperpixelRecoverFunction.apply

