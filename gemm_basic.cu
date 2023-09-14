#include<cuda.h>
#include<cuda_runtime.h>

#include "util.cuh"

namespace {
__global__ void gemmKernel(const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C,
                             unsigned M, unsigned N, unsigned K) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= N || r >= M) {
        return ;
    }
    Tensor2D<const float> tensorA = Tensor2D<const float>(A, M, K);
    Tensor2D<const float> tensorB = Tensor2D<const float>(B, K, N);
    Tensor2D<float> tensorC = Tensor2D<float>(C, M, N);
    float sum = 0.0;
    for (unsigned i = 0; i < K; i++) {
        sum += tensorA[r][i] * tensorB[i][c];
    }
    tensorC[r][c] = sum;
}
}

void gemmBasic(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
            unsigned M, unsigned N, unsigned K) {
    // Device malloc
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, M*K*sizeof(float));
    cudaMalloc(&d_y, K*N*sizeof(float));
    cudaMalloc(&d_z, M*N*sizeof(float));

    // Host to device
    cudaMemcpy(d_x, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // invoke
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N-1) / dimBlock.x + 1, (M-1) / dimBlock.y + 1);
    gemmKernel<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, M, N, K);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Device to host
    cudaMemcpy(C, d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}