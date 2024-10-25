# mpc_examples

This repo implements examples for the [MPC](https://github.com/RPS98/mpc) repository. The examples are implemented in Python and C++, and some of them use the multirotor simulator, both in Python [multirotor_simulator_py](https://github.com/RPS98/multirotor_simulator_py) and C++ [multirotor_simulator](https://github.com/RPS98/multirotor_simulator) versions.

## Installation

#### 1. Clone the repository

Clone this repository:

```bash
git clone hhttps://github.com/RPS98/mpc_examples.git
cd mpc_examples
git submodule update --recursive --init
```

#### 2. Follow MPC and multirotor simulator installation instructions

See [MPC](https://github.com/RPS98/mpc) [README](thirdparties/mpc/README.md) for acados installation, and [multirotor_simulator](https://github.com/RPS98/multirotor_simulator) [README](thirdparties/multirotor_simulator/README.md)

Remember to export **LD_LIBRARY_PATH** to acados and **PYTHONPATH** to acados and mpc, as detailed in [README](thirdparties/mpc/README.md).

#### 3. Build this repository

To build this repostory, follow the instructions from the root folder:

```bash
mkdir -p build
cd build
make -j4
```

## Example of the MPC using acados sim solver, both with Python and with C++

You can run the MPC using the Python interface or the C++ interface, from the root folder:

```bash
python3 examples/integrator_example.py
```

```bash
./build/mpc_examples_integrator_example
```

You can check the results in the `mpc_log.csv` file, and plot them with:

```bash
python3 examples/utils/sym_plot_results.py
```

## Example of the MPC using multirotor simulator, both with Python and with C++

You can run the MPC using the Python interface or the C++ interface, from the root folder:

```bash
python3 examples/multirotor_example.py
```

```bash
./build/mpc_examples_multirotor_example
```

You can check the results in the `ms_mpc_log.csv` file, and plot them with:

```bash
python3 examples/utils/ms_plot_resutls.py
```

## Development

If you want to change MPC model and test it, generate c code in thirdparties/mpc folder following [README](thirdparties/mpc/README.md) and then, rebuild this repository.