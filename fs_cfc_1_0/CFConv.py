import torch
import torch.nn as nn
from ControlledFilterConv import ControlledFilterConv

class CFConv( nn.Module):
    def __str__( self):
        return "CFConv( " + str( self.wch) + "(" + str( self.kx) + "x" + str( self.kx) + ")" + str( self.fch) + " -> " + str( self.och) + ")"

    __repr__ = __str__
    
    def __init__( self, wch, fch, och, kx, dtype = torch.float):
        super( CFConv, self).__init__()
        self.kx = kx;
        self.wch = wch;
        self.fch = fch;
        self.och = och;
        
        ks = kx * kx;

        self.weight = nn.Parameter( torch.zeros( [ks, och, fch + 1, wch + 1], dtype = dtype))
        
    def forward( self, inputw, inputf):
        if inputw.size(1) != self.wch:
            print( 'Wrong input channel: inputw is supposed to have ' + str( self.wch) + ' channels, but got ' + str( inputw.size(1)))
            exit(1)
        if inputf.size(1) != self.fch:
            print( 'Wrong input channel: inputf is supposed to have ' + str( self.fch) + ' channels, but got ' + str( inputf.size(1)))
            exit(1)
        if inputf.size(2) != inputw.size(2):
            print( 'Heights of input feature maps should match, but got ' + str( inputf.size(2)) + ' and ' + str( inputw.size(2)))
            exit(1)
        if inputf.size(3) != inputw.size(3):
            print( 'Width of input feature maps should match, but got ' + str( inputf.size(3)) + ' and ' + str( inputw.size(3)))
            exit(1)
        if self.weight.device.type != inputw.device.type:
            print( 'Wrong device: weight should be on ' + inputw.device.type + ' but got ' + self.weight.device.type)
            exit(1)
            
        return ControlledFilterConv( inputw, inputf, self.weight, self.och, self.kx)

    
