# Use CarNet in LAMMPS

This guide provides a detailed walkthrough for compiling LAMMPS with the `ML-IAP` (Machine Learning Interatomic Potentials) package to enable CarNet support. This setup uses the Python interface within ML-IAP to bridge LAMMPS with CarNet's PyTorch-based models.

## 1. System and Environment Setup

Before proceeding, ensure your system has a compatible **CUDA Driver** and a standard **MPI Library** (e.g., OpenMPI, MPICH). It is highly recommended to use a dedicated Conda environment.

```bash
conda create -n carnet-lammps python
conda activate carnet-lammps
```

## 2. Install Dependencies

### PyTorch
Install the version matching your CUDA driver following the [official PyTorch guide](https://pytorch.org/get-started/locally/).

### Other Dependencies
Install `cython`, `numpy`, and `cupy`. For `cupy`, ensure you install the version corresponding to your CUDA toolkit (e.g., `cupy-cuda12x` for CUDA 12.x). Refer to the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for more details.

```bash
pip install cython numpy cupy-cuda12x
```

## 3. Build LAMMPS

The build process requires specific flags to enable the Python/ML-IAP bridge and GPU acceleration via Kokkos.

### Load System Modules (HPC Environments)
If you are on an HPC cluster, load your compiler, CUDA, and MPI modules.

**Example:**
```bash
module load cmake cuda/12.8.0 openmpi/5.0.8
```

### Configure the Build

First, download the LAMMPS source code:

```bash
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
```

Next, navigate to the source directory and use a separate `build` directory:

```bash
mkdir build && cd build
cp ../cmake/presets/kokkos-cuda.cmake .
```

There are two ways to configure the build depending on your hardware access during compilation:

#### Method A: Compiling with GPU access (e.g., on a computing node)
If a GPU is available, Kokkos can automatically detect and configure the architecture.

```bash
cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D PKG_PYTHON=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  ../cmake
```

#### Method B: Compiling without GPU access (e.g., on a login node)
You must manually specify the target GPU architecture using `GPU_ARCH` and `Kokkos_ARCH`. For example, use `-D GPU_ARCH=sm_80 -D Kokkos_ARCH_AMPERE80=ON` for the Ampere architecture A100; use `-D GPU_ARCH=sm_120 -D Kokkos_ARCH_BLACKWELL120=ON` for RTX 5090.

To find the correct architecture for your GPU, refer to:
[LAMMPS Build Extras (Kokkos)](https://docs.lammps.org/Build_extras.html#kokkos)
and [LAMMPS Build Extras (GPU Package)](https://docs.lammps.org/Build_extras.html#gpu-package).

```bash
cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D PKG_PYTHON=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D GPU_ARCH=<YOUR_GPU_ARCH> \
  -D Kokkos_ARCH_<YOUR_KOKKOS_ARCH>=ON \
  ../cmake
```

### Compilation and Installation

Once configured, execute the following commands to build and install LAMMPS:

```bash
# Compile LAMMPS using 20 parallel processes
make -j 20

# Install the LAMMPS binary and libraries to the CMAKE_INSTALL_PREFIX directory
make install

# Install the 'lammps' Python package into the active conda environment
make install-python
```

## 4. Install CarNet

Finally, install the CarNet package itself from the source directory. 

```bash
# Navigate to the carnet source directory
cd <PATH_TO_CARNET_SOURCE>

# Install CarNet and its dependencies
pip install -e .
```

## 5. Test the Installation

To verify that the installation is successful and CarNet works with LAMMPS, please refer to the examples provided in the [example/](../example/) directory.

## Troubleshooting Architecture
A full list of supported GPU architectures can be found in the [LAMMPS Kokkos Documentation](https://docs.lammps.org/Build_extras.html#kokkos).
