from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polygon_inter_union_cuda',
    description='cuda implementation of computing polygon intersection and union',
    ext_modules=[CUDAExtension('polygon_inter_union_cuda',
        ['extensions.cpp', 'inter_union_cuda.cu'],
        extra_compile_args={
            'cxx': ['-std=c++14', '-O2', '-Wall'],
            'nvcc': [
                '-std=c++14', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall,-fno-gnu-unique',
                '-gencode=arch=compute_37,code=sm_37',
                '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86', '-gencode=arch=compute_86,code=compute_86'
            ],
        })
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch>=1.0.0a0',
        'torchvision',
        'pillow',
        'requests',
    ]
)
