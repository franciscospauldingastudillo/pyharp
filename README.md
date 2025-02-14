# pyharp

**pyharp** is a 1D general-purpose radiation solver based on the radiative transfer model **DISORT**. This code is intended for verification and validation of longwave (LW) and shortwave (SW) radiation components before their implementation in **CANOE**.

## Examples

Example implementations of shortwave and longwave radiation are provided in:

- `examples/amars_sw.cpp` (Shortwave radiation)
- `examples/amars_lw.cpp` (Longwave radiation)

**Note:** NetCDF is required for these examples to function correctly.

## Configuration and Compilation

To configure and build `pyharp`, run the following commands:

```sh
mkdir build
cd build
cmake .. -DNETCDF=ON
make -j8

