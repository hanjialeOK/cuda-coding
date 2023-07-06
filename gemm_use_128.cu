#include "util.cuh"

namespace {
__global__ void gemmKernel(const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C,
                             unsigned M, unsigned N, unsigned K) {
    constexpr unsigned kCount = sizeof(float4) / sizeof(float);
    unsigned c = (blockIdx.x * blockDim.x + threadIdx.x)*kCount;
    unsigned r = (blockIdx.y * blockDim.y + threadIdx.y)*kCount;
    if (c >= N || r >= M) {
        return ;
    }
    Tensor2D<const float> tensorA = Tensor2D<const float>(A, M, K);
    Tensor2D<const float4> tensorB = Tensor2D<const float4>(B, K, N/kCount);
    Tensor2D<float4> tensorC = Tensor2D<float4>(C, M, N/kCount);
    float4 sum[kCount] = {0};

    for (unsigned i = 0; i < K; i++) {
        float4 fragmentB = tensorB[i][c/kCount];
        for (unsigned j = 0; j < kCount; j++) {
            sum[j] += fragmentB * tensorA[r+j][i];
        }
    }
    for (unsigned i = 0; i < kCount; i++) {
        tensorC[r+i][c/kCount] = sum[i];
    }
}
}

void gemmUse128(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
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
    dim3 dimGrid((N/4-1) / dimBlock.x + 1, (M/4-1) / dimBlock.y + 1);
    gemmKernel<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, M, N, K);

    // Device to host
    cudaMemcpy(C, d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}