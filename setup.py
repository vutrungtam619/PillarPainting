from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pillarpainting_ops',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='build.voxel_op',
            sources=[
                'build/voxelization/voxelization.cpp',
                'build/voxelization/voxelization_cpu.cpp',
                'build/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='build.iou3d_op',
            sources=[
                'build/iou3d/iou3d.cpp',
                'build/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)