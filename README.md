# Spatial Distance Computations for PyTorch (SDCPT)

SDCPT is a tiny package that provides PyTorch-compatible spatial distance matrices computations.

### Supported Distance Computations

The current version (0.0.2) of SDCPT supports the following distance computations:

#### Distance Functions

- Jensen-Shannon Divergence Distance
- Directed Hausdorff Distance
- Bray-Curtis Distance

#### Validators

- Distance Matrix Validator

### Installation

Before installing SDCPT, ensure you meet the following prerequisites:

- Python >= 3.10
- PyTorch >= 2.0.0

If you don't have PyTorch 2.0.0 installed:

```bash
$ pip install torch==2.0.0
```

To install SDCPT, please first clone the remote repository and use the following pip command:

```bash
$ git clone https://github.com/SCALEDSL/SDCPT
$ cd SDCPT
$ pip install e .
```
