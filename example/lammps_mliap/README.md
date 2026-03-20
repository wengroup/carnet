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
```bash
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in lmp_input.in
```
