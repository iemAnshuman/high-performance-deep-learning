import os
import sys
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --- MPI Detection Logic (For Ring Reduce) ---
def get_mpi_flags():
    try:
        # Try to ask mpicc for the configuration
        show_c_flags = subprocess.check_output(["mpicc", "-showme:compile"], text=True).strip().split()
        show_ld_flags = subprocess.check_output(["mpicc", "-showme:link"], text=True).strip().split()
        
        include_dirs = [x[2:] for x in show_c_flags if x.startswith("-I")]
        library_dirs = [x[2:] for x in show_ld_flags if x.startswith("-L")]
        libraries = [x[2:] for x in show_ld_flags if x.startswith("-l")]
        return include_dirs, library_dirs, libraries
    except Exception as e:
        # Fallback for Apple Silicon (Homebrew) or just fail gracefully
        if sys.platform == "darwin" and os.path.exists("/opt/homebrew/include/mpi.h"):
            return ["/opt/homebrew/include"], ["/opt/homebrew/lib"], ["mpi"]
        print(f"[Warning] MPI compiler not found. Building 'ring_cpp' might fail.")
        return [], [], []

mpi_inc, mpi_lib_dirs, mpi_libs = get_mpi_flags()

# --- Define Extensions ---
extensions = [
    # 1. The Ring All-Reduce (Requires MPI)
    CppExtension(
        name='ring_cpp', 
        sources=['cpp_extensions/ring.cpp'],
        include_dirs=mpi_inc,
        library_dirs=mpi_lib_dirs,
        libraries=mpi_libs,
    ),
    # 2. The Quantization Packer (Pure C++)
    CppExtension(
        name='quantization_cpp',
        sources=['quantization/cpp_packing.cpp'],
        extra_compile_args=['-O3'] # Enable optimizations
    )
]

setup(
    name="hpc_dl_kernels",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension
    }
)