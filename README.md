# Beehive LevelZero JNI

Baremetal GPU and FPGA programming for Java using Intel's [Level Zero API](https://spec.oneapi.io/level-zero/latest/index.html). This project is a Java Native Interface (JNI) binding for Intel's Level Zero. This library is designed to be as closed as possible to the Level Zero API for C++. Subset of Level Zero 1.4.0 supported (Level Zero May 2022 version)


## Compilation & configuration of the JNI Level Zero API

### 1) Compile Level Zero API

#### Linux

Note: Using tag `v1.4.1` from `level-zero` which implements Level Zero Specification 1.2.

```bash
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
git checkout tags/v1.4.1
mkdir build
cd build
cmake ..
cmake --build . --config Release
cmake --build . --config Release --target package
```

#### Windows

Configuration:
- Lenovo IdeaPad Gaming 3 15IHU6
- Windows 11
- VS Community 2022
  + components C++, Git, Spectre mitigated libraries
- CMake 3.26.3, Maven 3.9.1, JDK 21

Run commands in _x64 Native Tools Command Prompt for VS 2022_.

```cmd
git clone https://github.com/oneapi-src/level-zero
cd level-zero
md build
cd build
cmake ..
cmake --build . --config Release
```

Note: Check for extisting Level Zero API libraries (e.g. `ze_tracing_layer.dll`) in `c:\windows\system32` if `zello_world.exe` fails.

```
rem check
.\bin\Release\zello_world.exe
```

### 2) Compile Level Zero JNI native code

Set the paths to the directory of Level Zero installation. Here are examples:

#### Linux

```bash
scl enable devtoolset-9 bash # << Only for CentOS
git clone https://github.com/beehive-lab/levelzero-jni
export ZE_SHARED_LOADER="<path-to-levelzero>/build/lib/libze_loader.so"
export CPLUS_INCLUDE_PATH=<path-to-levelzero>/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=<path-to-levelzero>/include:$CPLUS_INCLUDE_PATH
cd levelzero-jni/levelZeroLib
mkdir build
cd build
cmake ..
make
```

#### Windows

Note: Run commands in _x64 Native Tools Command Prompt for VS 2022_

```cmd
git clone https://github.com/otabuzzman/levelzero-jni
set ZE_SHARED_LOADER=%USERPROFILE%\lab\level-zero\build\lib\release\ze_loader.lib
set CPLUS_INCLUDE_PATH=%USERPROFILE%\lab\level-zero\include
set C_INCLUDE_PATH=%USERPROFILE%\lab\level-zero\include
cd levelzero-jni\levelZeroLib
md build
cd build
cmake ..
cmake --build . --config Release
```

#### Obtain a SPIR-V compiler (Linux and Windwos)

In case you want to compile kernels from OpenCL C to SPIR-V and use the Level Zero API, you need to download the `llvm-spirv` compiler. The implementation we are currently using is [Intel LLVM](https://github.com/intel/llvm).

### 3) Compile and run a Java test

#### Linux

```bash
mvn clean package
./scripts/run.sh   ## < This script compiles an OpenCL C program to SPIR-V using the llvm-spirv compiler (see 2.1)
```

The OpenCL C kernel provided for this example is as follows:

```c
__kernel void copyData(__global int* input, __global int* output) {
	uint idx = get_global_id(0);
	output[idx] = input[idx];
}
```

To compile from OpenCL C to SPIR-V:

```bash
clang -cc1 -triple spir copyData.cl -O0 -finclude-default-header -emit-llvm-bc -o copyData.bc
llvm-spirv copyData.bc -o copyData.spv
```

#### Windows

```cmd
mvn clean package

rem copyData.spv file expected in CWD
.\scripts\run.cmd

# more tests
.\scripts\copies.cmd
.\scripts\events.cmd
.\scripts\fences.cmd
.\scripts\kernelTimers.cmd
.\scripts\transferTimers.cmd
.\scripts\largeBuffers.cmd
```


## License

This project is developed at [The University of Manchester](https://www.manchester.ac.uk/), and it is fully open source under the [MIT](https://github.com/beehive-lab/levelzero-jni/blob/master/LICENSE) license.


## Acknowledgments

The work was partially funded by the EU Horizon 2020 [Elegant 957286](https://www.elegant-h2020.eu/) project, and [Intel Coorporation](https://www.intel.it/content/www/it/it/homepage.html).
