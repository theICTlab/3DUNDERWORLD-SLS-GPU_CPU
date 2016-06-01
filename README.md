# SLS
[![Build Status](https://travis-ci.org/v3c70r/SLS.svg?branch=dev)](https://travis-ci.org/v3c70r/SLS)

A Structured Light Scanner

## How to compile demo binaries

CMake is used as building system for this project. To buid the binaries, create a folder as bulding work directory.

```bash
mkdir build
cd build
cmake ..
make
```

Binaries are compiled and located in the `bin` folder. 

`SLS` is the reconstructor running on GPU

if CUDA is dected on your machine, a binary `SLS_GPU` will be compiled, which is a reconstructor running on GPU

`SLS_CALIB` is a manul calibration application.

`SLS_GRAYCODE` is an application to project graycode.

All of the binaries are designed to run with `alexandar` data.

## How to use the library

Please refer the code in `src/app/` to use your own data set.

## Known issues

Since there's no good API for cameras, the camera acquisition is not implemented. However, interfaces are provided.
We welcome you to implement your camera class and make a pull request to this project.






