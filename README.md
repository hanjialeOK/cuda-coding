# CUDA-coding

## Introduction

学习cuda的一些指南

- [https://zhuanlan.zhihu.com/p/34587739](https://zhuanlan.zhihu.com/p/34587739)
- [https://developer.nvidia.com/zh-cn/blog/cuda-programming-model-interface-cn/](https://developer.nvidia.com/zh-cn/blog/cuda-programming-model-interface-cn/)

## VSCODE

vscode配置，快捷键 `ctrl+shift+p` 然后输入 `config` 来编辑 json 文件

加上cuda头文件的路径

```c
"includePath": [
    "${workspaceFolder}/**",
    "/usr/local/cuda/include",
],
```

修改编译工具路径

```c
"compilerPath": "/usr/local/cuda/bin/nvcc",
```

增加并修改 tasks.json.

```c
"args": [
    // "-fdiagnostics-color=always",
    "-g",
    "${file}",
    "-o",
    "${fileDirname}/${fileBasenameNoExtension}",
    "${fileDirname}/gemm_basic.cu",
    "${fileDirname}/gemm_use_128.cu",
    "${fileDirname}/gemm_use_tile.cu",
    "${fileDirname}/gemm_use_128_openmlsys.cu",
```

## Requirements

安装 GDB 工具

```c
apt install build-essential gdb
```

vscode 安装 `Nsight Visual Studio Code Edition` 和 `C/C++` 插件

## Debug

```c
nvcc -g -G -o test main.cu gemm_basic.cu gemm_use_128.cu gemm_use_tile.cu gemm_use_128_openmlsys.cu
```

或者

```c
make DEBUG=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

## Run

```c
nvcc -o test main.cu gemm_basic.cu gemm_use_128.cu gemm_use_tile.cu gemm_use_128_openmlsys.cu
./test
```

或者

```c
make NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```
