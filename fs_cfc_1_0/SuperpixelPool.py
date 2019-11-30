import torch
import torch.nn as nn
from torch.autograd import Function

import spl_impl

class SuperpixelPoolFunction( Function):
    @staticmethod
    def forward( ctx, inputsp, inputf, K):
        ctx.save_for_backward( inputsp, inputf)
        ctx.K = K
        return spl_impl.forward( inputsp, inputf, K)

    @staticmethod
    def backward( ctx, grad_output):
        inputsp, inputf = ctx.saved_tensors
        K = ctx.K
        gradinputsp, gradinputf = spl_impl.backward( inputsp, inputf, grad_output, K)
        return gradinputsp, gradinputf, None

SuperpixelPool = SuperpixelPoolFunction.apply

