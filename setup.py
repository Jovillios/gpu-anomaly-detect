from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rolling_mean_cuda",
    ext_modules=[
        CUDAExtension(
            name="rolling_mean_cuda",
            sources=["rolling_mean.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
