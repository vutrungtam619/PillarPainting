from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pillarpainting_ops',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='package.voxel_op',
            sources=[
                'package/voxelization/voxelization.cpp',
                'package/voxelization/voxelization_cpu.cpp',
                'package/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='package.iou3d_op',
            sources=[
                'package/iou3d/iou3d.cpp',
                'package/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)