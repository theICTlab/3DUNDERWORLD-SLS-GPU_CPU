# 3DUNDERWORLD-Structured-Light-Scanner
[![Build Status](https://travis-ci.org/theICTlab/3DUNDERWORLD-SLS-GPU_CPU.svg?branch=dev)](https://travis-ci.org/theICTlab/3DUNDERWORLD-SLS-GPU_CPU)

Structured light scanner is a tool to reconstruct point cloud from series of images lit by patterned light. As showing in the following image, patterns are projected to the object in the grid manner. Each cell in the grid has a unique sequence. The corresponding points can be extracted by decoding the sequences from both images.
![img](https://raw.githubusercontent.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/dev/screenshots/flow.png)

In this version, the reconstruction process is reimplemented both on CPU and GPU. Both versions accelerate the reconstruction time. Especially the later one. 


## Dependencies
* OpenCV2
* CUDA (6.0+)
* glm

## Howto
### How it works

The main building blocks of the libraries are `ImageProcessor` and `Reconstructor`. Image processors take images, projector parameters and camera calibration parameters as input and produce `Buckets`. `Reconstructor` combines all the Buckets to generate the point cloud.  
![Alt text](https://rawgit.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/dev/doc/how-it-works.svg)



### Compile and run demo binaries
Demo binaries and data is included.  A CMake script is included to compile the binaries and libraries. 
```bash
git clone --recursive https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU.git
cd 3DUNDERWORLD-SLS-GPU_CPU
mkdir build
cd build
cmake ..
make
```
If CUDA is detected in your system, the cmake flag `ENABLE_CUDA` is set to `on` by default, both CPU and GPU constructors are created in the `bin` folder. Otherwise, only CPU constructor will be generated. To run the binaries, you can pass the data folder and camera configuration files as parameters, or using the default value if you compiled with the commands above.

You can use `cmake .. -DENABLE_CUDA=off` to disable CUDA if you don't want compile the GPU binary.

To build the GPU reconstructor for AMD GPUs instead of NVIDIA, configure with `-DUSE_HIP=ON` and set the target architecture via `CMAKE_HIP_ARCHITECTURES` (for example `gfx90a` for CDNA2 / MI200, `gfx942` for CDNA3 / MI300, or `gfx1100` for RDNA3). This requires a [ROCm](https://rocm.docs.amd.com/) installation providing HIP; the build was tested with ROCm 7.2.
```
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make
```
If CMake does not pick up the HIP compiler from the ROCm installation, name it explicitly, for example `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`. Leaving `CMAKE_HIP_ARCHITECTURES` unset makes CMake detect the architecture of the GPU in the machine, which fails on a machine without one, so build machines should set it. The default `ENABLE_CUDA` (NVIDIA) path is unchanged.

On Windows the prebuilt OpenCV packages install their headers as `include/opencv2/...`, while the sources include them as `<opencv4/opencv2/...>` (the layout of the Linux distribution packages). Set `OPENCV4_COMPAT_DIR` to a directory that contains an `opencv4` entry resolving to the OpenCV include root, so that both spellings work: with a junction created by `mklink /J C:\opencv-compat\opencv4 C:\opencv\build\include`, configure with `-DOPENCV4_COMPAT_DIR=C:/opencv-compat`.

### Enable test
A test using [Google Test Framework](https://github.com/google/googletest.git) is included in the build. To use the test, enable `GTEST` flag at configuration stage.
```bash
git clone https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU.git
cd 3DUNDERWORLD-SLS-GPU_CPU
mkdir build
cd build
cmake .. -DGTEST=ON
make
make test
```
Google test will be downloaded and compiled during the compilation stage. 

### Build documentation
Document with Doxygen is availabe to be built. Use `-DBUILD_DOC=on` cmake flag to enable documentation. Run `make doc`, the documentation will be generated in the `doc` directory of building path. 

### Code coverage
Code coverage report is provided by using [lcov](http://ltp.sourceforge.net/coverage/lcov.php). In order to get the code coverage report, enable both test and coverage flag: `-DGTEST=on -DCOVERAGE=on` and run `make coverage`. The coverage report will be generated in the `coverage` directory of building path. 

### Run demo binary
```bash
usage: ./SLS --leftcam=string --rightcam=string --leftconfig=string --rightconfig=string --output=string --format=string --width=unsigned long --height=unsigned long [options] ... 
options:
  -l, --leftcam        Left camera image folder (string)
  -r, --rightcam       Right camera image folder (string)
  -L, --leftconfig     Left camera configuration file (string)
  -R, --rightconfig    Right camera configuration file (string)
  -o, --output         Output folder (string)
  -f, --format         Suffix of image files, e.g. jpg (string)
  -w, --width          Projector width (unsigned long)
  -h, --height         Projector height (unsigned long)
  -?, --help           print this message

```
The configuration files are `xml` files generated by OpenCV calibration tool, including intrinsic and extrinsic parameters.

Note: --output is the name and location of the output file (either ply ot obj) e.g ./output/output.ply

### Grab the output
Most of the outputs are written to the log file named `SLS.log`. To track the outputs while running the binary, run `tail -f sls.log` in another terminal of the same directory. 

### Use the library

This project also provides libraries that can be easily integrated into other projects. Here's a quick start code for using the libraries. An example of [CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/blob/dev/src/app/App.cpp) and [GPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/blob/dev/src/app/App_CUDA.cu) applications are included in the repository. 

### Run library on demo data
Run the library using the included example images

```bash
cd 3DUNDERWORLD-SLS-GPU_CPU
mkdir test
cd test
mkdir Output
cp -R ../data/alexander/leftCam ./
cp -R ../data/alexander/rightCam ./
cmake ..
make
./bin/SLS --leftcam=./leftCam/dataset1 --rightcam=./rightCam/dataset1 --leftconfig=./leftCam/calib/output/calib.xml --rightconfig=./rightCam/calib/output/calib.xml --output=./Output/output.ply --format=jpg --width=1024 --height=768

```

## Known issues

Since there's no good API for cameras, the camera acquisition is not implemented. However, interfaces are provided.
We welcome you to implement your camera class and make a pull request to this project.

Also please note that since the example data files are currently included in the repository, the size of repository is 300MB. It may take while to clone. 

## Version Information

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

```
@article{GuHerakleousPoullis3dunderworld,
  title={A Structured-Light Scanning Software for Rapid Geometry Acquisition},
  author={Gu, Qing and Herakleous, Kyriakos and Poullis, Charalambos},
  journal={TBA},
  year={2016}
}

@article{DBLP:journals/corr/HerakleousP14,
  author    = {Kyriakos Herakleous and
               Charalambos Poullis},
  title     = {3DUNDERWORLD-SLS: An Open-Source Structured-Light Scanning System
               for Rapid Geometry Acquisition},
  journal   = {CoRR},
  volume    = {abs/1406.6595},
  year      = {2014},
  url       = {http://arxiv.org/abs/1406.6595},
  timestamp = {Tue, 01 Jul 2014 11:58:08 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/HerakleousP14},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
