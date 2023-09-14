#include "util.cuh"

namespace {
__global__ void gemmKernel(const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C,
                             unsigned M, unsigned N, unsigned K) {
    constexpr unsigned kCount = sizeof(float4) / sizeof(float);
    unsigned c = (blockIdx.x * blockDim.x + threadIdx.x)*kCount*2;
    unsigned r = (blockIdx.y * blockDim.y + threadIdx.y)*kCount*2;
    if (c >= N || r >= M) {
        return ;
    }
    Tensor2D<const float> tensorA = Tensor2D<const float>(A, M, K);
    Tensor2D<const float4> tensorB = Tensor2D<const float4>(B, K, N/kCount);
    Tensor2D<float4> tensorC = Tensor2D<float4>(C, M, N/kCount);
    float4 sum[2][2][kCount] = {0};

    for (unsigned k = 0; k < K; k++) {
        for (unsigned iterA = 0; iterA < 2; iterA++) {
            float fragmentA[kCount];
            for (unsigned offset = 0; offset < kCount; offset++) {
                fragmentA[offset] = tensorA[r+offset+kCount*iterA][k];
            }
            for (unsigned iterB = 0; iterB < 2; iterB++) {
                float4 fragmentB = tensorB[k][c/kCount+iterB];
                for (unsigned j = 0; j < kCount; j++) {
                    sum[iterA][iterB][j] += fragmentB * fragmentA[j];
                }
            }
        }
    }
    for (unsigned iterA = 0; iterA < 2; iterA++) {
        for (unsigned iterB = 0; iterB < 2; iterB++) {
            for (unsigned j = 0; j < kCount; j++) {
                tensorC[r+j+kCount*iterA][c/kCount+iterB] += sum[iterA][iterB][j];
            }
        }
    }
}
}

void gemmUseTile(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
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
    dim3 dimGrid((N/8-1) / dimBlock.x + 1, (M/8-1) / dimBlock.y + 1);
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