import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

modules = [
    CppExtension(
        'roi_align.crop_and_resize_cpu',
        ['roi_align/src/crop_and_resize.cpp'],
        extra_compile_args={'cxx': ['-g', '-fopenmp']}
        )
]

if torch.cuda.is_available():
    modules.append(
        CUDAExtension(
            'roi_align.crop_and_resize_gpu',
            ['roi_align/src/crop_and_resize_gpu.cpp',
             'roi_align/src/cuda/crop_and_resize_kernel.cu'],
            extra_compile_args={'cxx': ['-g', '-fopenmp'],
                                'nvcc': ['-O2']}
        )
    )

setup(
    name='roi_align',
    version='0.0.2',
    description='PyTorch version of RoIAlign',
    author='Long Chen',
    author_email='longch1024@gmail.com',
    url='https://github.com/longcw/RoIAlign.pytorch',
    packages=find_packages(exclude=('tests',)),

    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=1.2.0']
)
