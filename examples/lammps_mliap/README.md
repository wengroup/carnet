# CarNet LAMMPS ML-IAP Example

This directory contains an example of using CarNet in LAMMPS.

## Files
- `export_lammps_mliap.py`: Script to convert a trained CarNet model for LAMMPS ML-IAP.
- `lmp_input.in`: LAMMPS input file.
- `water.data`: LAMMPS data file.

## How to Run

### 1. Create a LAMMPS-Compatible Model
Update `model_path` in `export_lammps_mliap.py` to point to a trained CarNet model and run `python export_lammps_mliap.py`. This generates a `.pt` model file.

### 2. Configure LAMMPS
Update the `pair_style` in `lmp_input.in` to point to the exported `.pt` file.

### 3. Run LAMMPS
Run LAMMPS using the following command (adjust the path to your `lmp` executable as needed):

#### Using a single GPU
```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in lmp_input.in
```

#### Using multiple GPUs (one GPU per MPI rank)

```bash
# Request 2 (or more) GPUs
mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in lmp_input.in
```
Note:

If you get a `Segmentation fault` error, and have warnings like `WARNING:
PackForwardComm has only been tested on Kokkos devices
(src/KOKKOS/pair_mliap_kokkos.cpp:445)`, this means direct GPU-to-GPU communication is
not enabled in your MPI. The `mliap/kk` pair style exchanges ghost-atom data directly
between GPUs, which requires a CUDA-aware MPI. 

For example, with OpenMPI you can confirm CUDA support with:
```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
# should report: ...:value:true
```

If it reports `false`, rebuild your MPI with CUDA support. Once a CUDA-aware MPI is
used, the direct GPU-to-GPU path is enabled and the run will work.
