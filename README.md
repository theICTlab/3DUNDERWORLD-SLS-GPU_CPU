# 3DUNDERWORLD-Structured-Light-Scanner
=====================================

### [v4] 
Lead software developer: Qing Gu
     
Researchers: Qing Gu, Charalambos Poullis
     
Immersive and Creative Technologies Lab (http://www.theICTlab.org), Concordia University

### [v1-v3.2] 

Lead software developer: Kyriakos Herakleous

Researchers: Kyriakos Herakleous, Charalambos Poullis

Immersive and Creative Technologies Lab (http://www.theICTlab.org), Concordia University


#### Part of the 3DUNDERWORLD project: http://www.3dunderworld.org

##### IMPORTANT: To use this software, YOU MUST CITE the following in any resulting publication:*

@article{GuHerakleousPoullis3dunderworld,
  title={A Structured-Light Scanning Software for Rapid Geometry Acquisition},
  author={Gu, Qing and Herakleous, Kyriakos and Poullis, Charalambos},
  journal={TBA},
  year={2016}
}

@article{herakleous20143dunderworld,
  title={3DUNDERWORLD-SLS: An Open-Source Structured-Light Scanning System for Rapid Geometry Acquisition},
  author={Herakleous, Kyriakos and Poullis, Charalambos},
  journal={arXiv preprint arXiv:1406.6595},
  year={2014}
}


[![status](http://joss.theoj.org/papers/4329bcbc7bba33961a5e749dcacb995b/status.svg)](http://joss.theoj.org/papers/4329bcbc7bba33961a5e749dcacb995b)


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

All of the binaries are designed to run with [`alexandar`](https://github.com/v3c70r/SLS/tree/dev/data/alexander) data.

## How to use the library

Please refer to the code in [`src/app/`](https://github.com/v3c70r/SLS/tree/dev/src/app) to use your own data set.

## Known issues

Since there's no good API for cameras, the camera acquisition is not implemented. However, interfaces are provided.
We welcome you to implement your camera class and make a pull request to this project.

Also please note that since the example data files are currently included in the repository, the size of repository is 300MB. It may take while to clone. 
