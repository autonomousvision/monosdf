from distutils.command.build import build
import os
from torch.utils.cpp_extension import load
from pathlib import Path

Path('./tmp_build/').mkdir(parents=True, exist_ok=True)

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_hash_encoder',
                extra_cflags=['-O3', '-std=c++14'],
                extra_cuda_cflags=[
                    '-O3', '-std=c++14', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory='./tmp_build/',
                verbose=True,
                )

__all__ = ['_backend']