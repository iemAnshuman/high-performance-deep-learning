from setuptools import setup, find_packages

setup(
    name="hpc_dl_kernels",
    version="0.1.0",
    description="High-Performance Triton Kernels & Quantization for Physics-Informed AI",
    author="Anshuman Agrawal",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0",
        "numpy",
        "pytest",
        "scipy"
    ],
)
