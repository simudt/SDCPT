# Distance Computations for PyTorch Functions (DCPF)

DCPF is a tiny package that provides spatial functions for computing and validating PyTorch compatible distance matrices computations.

### Supported Distance Computations

The current version (0.0.1) of DCPF supports the following distance computations:

- Jensen-Shannon Divergence Distance
- Directed Hausdorff Distance
- Distance Matrix Validator

### Installation

Before installing DCPF, ensure you meet the following prerequisites:

- Python >= 3.10
- PyTorch >= 2.0.0

If you don't have PyTorch 2.0.0 installed:

```bash
$ pip install torch==2.0.0
```

To install DCPF, please first clone the remote repository and use the following pip command:

```bash
$ git clone https://github.com/simugrad/DCPF
$ cd DCPF
$ pip install e .
```
