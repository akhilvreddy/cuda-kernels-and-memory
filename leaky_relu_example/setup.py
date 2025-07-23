from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='leaky_relu_cuda',
    ext_modules=[
        CUDAExtension(
            name='leaky_relu_cuda',
            sources=['leaky_relu_bindings.cpp', 'leaky_relu_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)