#--INTRODUCTION--#

This directory contains the implementations of Superpixel Pooling Layers,
Superpixel Recovering Layers & Controlled Filter Convolution layers. All of
the files should be used under the PyTorch environment.

#--FILE DESCRIPTIONS--#

\

\module_impl		Low level implementations

\module_impl\*.cpp	C++ interface definition files
\module_impl\*.cu		CUDA implementation files

\module_setup.py	Module setup script for PyTorch
\CFConv.py		Python interface of CFC layers

\*.py			PyTorch function definition files

#--USAGE--#

1. Setup up the modules by simply using module_setup.py.

2. To use Superpixel Pooling layers and Superpixel Recovering layers, 
import the functions (under PyTorch definition) as follows:
	from SuperpixelPool import SuperpixelPool
	from SuperpixelRecover import SuperpixelRecover
, and then use SuperpixelPool and SuperpixelRecover as any other 
functions in torch.nn.functional, PS. SuperpixelPool.py & 
SuperpixelRecover.py are in this directory.

3. To use Controlled Filter Convolution layers, first import CFConv module 
as follows:
	from CFConv import CFConv
, and create instances of CFConv with the following parameters:
	wch	Size of the 2nd dim (the order of dims is like what  
		Conv2d requires) of inputw (the 1st feature map)
	fch	Size of the 2nd dim (the order of dims is like what  
		Conv2d requires) of inputf (the 2nd feature map)
	och	Size of the 2nd dim (the order of dims is like what  
		Conv2d requires) of output
	kx	Size of the kernel will be kx X kx, i.e., ky equals to kx 
		by default
	dtype	Data type of the feature maps. Can be torch.float or 
		torch.double.
Strides of both directions are fixed to 1 in this implementation.

4. In forward runs, CFConv layers accept 2 parameters inputw and inputf, 
which represent the 2 input feature maps of CFC respectively, instead of a 
single input feature map like most of the other modules.

