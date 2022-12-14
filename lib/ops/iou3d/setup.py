import setuptools
from setuptools import setup
print(setuptools.__version__)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d',
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
