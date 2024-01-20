import torch
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

file_lis = [
    './src/scatter_points_cpu.cpp',
    './src/scatter_points_cuda.cu',
    './src/voxelization.cpp',
    './src/voxelization_cpu.cpp',
    './src/voxelization_cuda.cu'
] 

extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp', '-DWITH_CUDA'],
    'nvcc': ['-O3', '-DWITH_CUDA']
}

extension_type = CUDAExtension
setup(
    name='voxel_layer',
    packages=find_packages(),
    ext_modules=[
        extension_type('voxel_layer',
                       file_lis,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
