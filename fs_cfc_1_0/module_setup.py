from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='fscfc',
      ext_modules=[ CUDAExtension('spl_impl', ['module_impl/SuperpixelPool.cpp', 'module_impl/SuperpixelPool_impl.cu']),
                    CUDAExtension('srl_impl', ['module_impl/SuperpixelRecover.cpp', 'module_impl/SuperpixelRecover_impl.cu']),
                    CUDAExtension('cfc_impl', ['module_impl/ControlledFilterConv.cpp', 'module_impl/ControlledFilterConv_impl.cu']),
      ],
      cmdclass={'build_ext': BuildExtension})


