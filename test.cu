#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<memory.h>

using namespace std;

const int N = 128;

struct Matrix {
    int width;
    int height;
    float* elements;
    size_t pitch;
    size_t DeviceMalloc(int, int);
    void DeviceFree();
    __host__ __device__ float* operator[] (size_t row) {
        return (float*)((char*)elements + row * pitch);
    }
};

size_t Matrix::DeviceMalloc(int w, int h) {
    size_t nBytes = w * h * sizeof(float);
    width = w;
    height = h;
    cudaMallocPitch(&elements, &pitch, width*sizeof(float), height);
    return nBytes;
}

void Matrix::DeviceFree() {
    cudaFree(elements);
}

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    C[c][r] = A[c][r] + B[c][r];
}

__device__ float Get(float *d_x, int r, int c, size_t pitch) {
    float* row = (float*)((char*)d_x + r * pitch);
    return row[c];
}

__device__ void Set(float *d_x, int r, int c, size_t pitch, float val) {
    float* row = (float*)((char*)d_x + r * pitch);
    row[c] = val;
}

__global__ void MatAddKernel(float *d_x, float *d_y, float *d_z, size_t pitch) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    float data_x = Get(d_x, r, c, pitch);
    float data_y = Get(d_y, r, c, pitch);
    Set(d_z, r, c, pitch, data_x + data_y);
}

__global__ void MatMulKernel(float *d_x, float *d_y, float *d_z, size_t pitch) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        float data_x = Get(d_x, r, i, pitch);
        float data_y = Get(d_y, i, c, pitch);
        sum += data_x * data_y;
    }
    Set(d_z, r, c, pitch, sum);
}

void init(float A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1;
        }
    }
}

void show(float A[N][N], int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    // Get basic info
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    cout << "使用GPU device " << dev << ": " << devProp.name << endl;
    cout << "SM的数量：" << devProp.multiProcessorCount << endl;
    cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
    cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
    cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << endl;
    cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << endl;

    // Allocate memory in host
    float A[N][N], B[N][N], C[N][N];
    init(A);
    init(B);
    memset(&C[0][0], 0, sizeof(C));
    cout << "init C:" << endl;
    show(C, 10);

    // Allocate memory in device
    float *d_x, *d_y, *d_z;
    size_t pitch;
    cudaMallocPitch(&d_x, &pitch, N*sizeof(float), N);
    cudaMallocPitch(&d_y, &pitch, N*sizeof(float), N);
    cudaMallocPitch(&d_z, &pitch, N*sizeof(float), N);

    // Copy data from host to device
    cudaMemcpy2D(d_x, pitch, &A[0][0], N*sizeof(float), N*sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_y, pitch, &B[0][0], N*sizeof(float), N*sizeof(float), N, cudaMemcpyHostToDevice);

    // Invoke
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatMulKernel<<<numBlocks, threadsPerBlock>>>(d_x, d_y, d_z, pitch);

    // Copy the result from device to host
    cudaMemcpy2D(&C[0][0], pitch, d_z, N*sizeof(float), N*sizeof(float), N, cudaMemcpyDeviceToHost);

    cout << "output:" << endl;
    show(C, 10);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}