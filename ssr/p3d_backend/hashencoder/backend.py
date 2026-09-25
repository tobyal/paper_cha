from distutils.command.build import build
import os
from torch.utils.cpp_extension import load
from pathlib import Path

_build_path = Path(__file__).resolve().parents[3] / '.build' / 'hashencoder'
_build_path.mkdir(parents=True, exist_ok=True)

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_hash_encoder',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=[
                    '-O3', '-std=c++17', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory=str(_build_path),
                verbose=True,
                )

__all__ = ['_backend']
