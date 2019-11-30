import torch
import torch.nn as nn
from torch.autograd import Function

import cfc_impl

class ControlledFilterConvFunction( Function):
    @staticmethod
    def forward( ctx, inputw, inputf, paramw, nplane_o, kx):
        ctx.save_for_backward( inputw, inputf, paramw)
        ctx.nplane_o = nplane_o
        ctx.kx = kx
        output = cfc_impl.forward( inputw, inputf, paramw, nplane_o, kx, kx)
        return output

    @staticmethod
    def backward( ctx, grad_output):
        inputw, inputf, paramw = ctx.saved_tensors;
        kx = ctx.kx;
        nplane_o = ctx.nplane_o
        gradinputw, gradinputf, gradparamw = cfc_impl.backward( inputw, inputf, paramw, grad_output, kx, kx)
        return gradinputw, gradinputf, gradparamw, None, None

ControlledFilterConv = ControlledFilterConvFunction.apply
