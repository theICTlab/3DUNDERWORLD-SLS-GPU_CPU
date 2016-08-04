---
  title: 'A Structured-Light Scanning Software for Rapid Geometry Acquisition'
  tags:
    - 3D
    - structured light scanning
    - reconstruction
    - scanning
  authors:
   - name: Qing Gu
     affiliation: Immersive and Creative Technologies Lab, Concordia University
   - name: Kyriakos Herakleous
     affiliation: Immersive and Creative Technologies Lab, Concordia University
   - name: Charalambos Poullis
     orcid: 0000-0001-5666-5026
     affiliation: Immersive and Creative Technologies Lab, Concordia University
  date: 27 July 2016
  bibliography: paper.bib
---
# Summary
This is a 3D scanning system which is based on the principle of structured-light. We introduce our open-source scanning software system "3DUNDERWORLD-SLS" which implements the techniques both on CPU and GPU. We have performed extensive testing with a wide range of models and the results are documented in our report. 

[![Build Status](https://travis-ci.org/v3c70r/SLS.svg?branch=dev)](https://travis-ci.org/v3c70r/SLS)


## How to compile demo binaries

CMake is used to build this project. To buid the binaries, create a folder as bulding work directory.

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

All of the binaries are designed to run with [`alexander`](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/blob/dev/data/alexander) data.

## How to use the library

Please refer to the code in [`src/app/`](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/blob/dev/src/app) to use your own data set.

## Known issues

Since there's no good API for cameras, the camera acquisition is not implemented. However, interfaces are provided.
We welcome you to implement your camera class and make a pull request to this project.

