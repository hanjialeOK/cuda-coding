#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

#include<util.cuh>

using namespace std;

const unsigned M = 1024;
const unsigned N = 1024;
const unsigned K = 2048;

void init(float *A, int w, int h, int val=1) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            A[i*w+j] = rand() % 5;
        }
    }
}

void show(float *A, int w, int h, int size=10) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << A[i*w+j] << " ";
        }
        cout << endl;
    }
}

__global__ void gemmKernel_basic(const float *__restrict__ A,
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

__global__ void gemmKernel_float4(const float *__restrict__ A,
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
    float4 sum[kCount];

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

__global__ void gemmKernel_tile(const float *__restrict__ A,
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
    float4 sum[2][2][kCount];

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

void matMul(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
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
    gemmKernel3<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, M, N, K);

    // Device to host
    cudaMemcpy(C, d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

float nativeMatMul(float *A, float *B, float *C) {
    float diff = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K+k] * B[k*N+j];
            }
            diff += abs(C[i*N+j] - sum);
        }
    }
    return diff;
}

int main() {
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

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float *A, *B, *C;
    A = (float*)malloc(M*K*sizeof(float));
    B = (float*)malloc(K*N*sizeof(float));
    C = (float*)malloc(M*N*sizeof(float));
    srand(time(0));
    init(A, M, K, 1);
    init(B, K, N, 2);
    cout << "init C:" << endl;
    show(C, M, N);

    cudaEventRecord(startEvent);
    matMul(A, B, C, M, N, K);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    cout << "output C:" << endl;
    show(C, M, N);

    float diff = nativeMatMul(A, B, C);
    cout << "total err: " << diff << endl;

    printf("Average Time: %.3f ms\n", milliseconds);

    free(A);
    free(B);
    free(C);
}