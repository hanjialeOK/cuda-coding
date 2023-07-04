# CUDA-coding

## Intro

Some instructions for learning cuda:

- [https://zhuanlan.zhihu.com/p/34587739](https://zhuanlan.zhihu.com/p/34587739)
- [https://developer.nvidia.com/zh-cn/blog/cuda-programming-model-interface-cn/](https://developer.nvidia.com/zh-cn/blog/cuda-programming-model-interface-cn/)

## Settings

### edit configurations

For vscode, `ctrl+shift+p` and type `config` to edit configurations(json).

Add `includePath` in c_cpp_properties.json.

```c
"includePath": [
    "${workspaceFolder}/**",
    "/usr/local/cuda/include",
],
```

change `compilerPath` in c_cpp_properties.json.

```c
"compilerPath": "/usr/local/cuda/bin/nvcc",
```

Add these in settings.json.

```c
{
    "files.associations": {
        "*.cu": "cpp",
        "*.cuh": "cpp",
    }
}
```

### debugging

Install gdb tools

```c
apt install build-essential gdb
```

Open .cu file and press the play button in the top right corner of the editor.

## Run

```c
nvcc -o test test.cu
./test
```
