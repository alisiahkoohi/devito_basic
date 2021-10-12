# A basic devito modeling example + zero-offset gather

## Prerequisites

First clone the repository:

```bash
git clone https://github.com/alisiahkoohi/devito_basic
cd devito_basic/
```

Follow the steps below to install the necessary libraries.

```bash
conda env create -f environment.yml
conda activate devito
```

Include the following environment variables in `~/.bashrc` to maximize the utilization of your hardware.

```bash
export DEVITO_ARCH=gnu
export DEVITO_LANGUAGE=openmp
export OMP_NUM_THREADS=8 # Adjust depending on your architecture.
```

## Example

To run a basic seismic modeling example, run:

```bash
python modeling.py
```

and for creating a zero-offset gather on the same model, execute:

```bash
python zero_offset.py
```

The created zero-offset gather will be saved in a HDF5 file.

## Author

Ali Siahkoohi
