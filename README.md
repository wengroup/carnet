# CarNet

Cartesian Natural Tensor Networks - atomistic machine learning for interatomic
potentials, scalar properties, tensorial properties and beyond.

## Installation

It is recommended to create a virtual environment first (e.g. using conda) before installing.

**1. Install PyTorch**

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally)
to install PyTorch for your platform and CUDA version.

**2. Install `natt`**

```bash
pip install git+https://github.com/wengroup/natt.git
```

**3. Clone and install CarNet**

```bash
git clone https://github.com/wengroup/carnet.git
cd carnet
pip install -e .
```

(Optional) To include dependencies for testing:

```bash
pip install -e ".[test]"
```

## Features

- **Interatomic potentials**: train machine learning interatomic potentials with interfaces for:
  - [ASE](https://wiki.fysik.dtu.dk/ase) via a calculator
  - [LAMMPS](https://www.lammps.org) via the [MLIAP](https://docs.lammps.org/Build_extras.html#ml-iap) package
- **Tensorial properties**: predict tensorial properties (e.g. polarizability, dielectric tensors, elastic tensors) of molecules and materials


## Examples
See the [examples](./examples) directory for more details on how to use CarNet in ASE and LAMMPS.

## Citation
Chen, Q., Pattamatta, A.S.L., Wang, B., Srolovitz, D.J. and Wen, M., 2026. Atomistic Machine Learning with Irreducible Cartesian Natural Tensors. arXiv preprint arXiv:2510.04015.

```latex
@article{chen2026atomistic,
  title   = {Atomistic Machine Learning with Irreducible Cartesian Natural Tensors},
  author  = {Chen, Qun and Pattamatta, ASL and Wang, Boyu and Srolovitz, David J and Wen,
  Mingjian},
  journal = {arXiv preprint arXiv:2510.04015},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.04015},
}
```