pyharp is a 1D general-purpose radiation solver based on the RT model: DISORT. The purpose of this code is for verification/validation of LW and SW components of radiation before their implementation in CANOE.

Examples of shortwave and longwave radiation are in:

examples/amars_sw.cpp
examples/amars_lw.cpp

Note that netcdf is required for this example to work.

To configure:
mkdir build
cd build
cmake .. -DNETCDF=ON
make -j8

If successful, two executable files will be placed in build/bin

amars_lw.release for longwave
amars_sw.release for shortwave

All credit and thanks to Cheng Li (https://github.com/chengcli/pyharp).
