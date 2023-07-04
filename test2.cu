#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std;

const unsigned M = 512;
const unsigned N = 512;
const unsigned K = 1024;

template <typename T>
struct __device_builtin__ Tensor2D {
    unsigned height;
    unsigned width;
    T *const data_ptr;
    template <typename t>
    __host__ __device__ Tensor2D(t &&ptr, unsigned h, unsigned w)
        : data_ptr(reinterpret_cast<T *>(ptr)), height(h), width(w) {}
    __host__ __device__ T * operator[](unsigned row) const {
        return &data_ptr[row*width];
    }
    __host__ __device__ T & operator()(unsigned row, unsigned col) const {
        return data_ptr[row*width + col];
    }
};

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

void init(float *A, int w, int h, int val=1) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            A[i*w+j] = rand() % 10;
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

__global__ void MatMulKernel1(const float *__restrict__ A,
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

__global__ void MatMulKernel(const float *__restrict__ A,
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

__global__ void MatMulKernel3(const float *__restrict__ A,
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

void MatMul(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
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
    MatMulKernel3<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, M, N, K);

    // Device to host
    cudaMemcpy(C, d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

float nativeMatMul(float A[M][K], float B[K][N], float C[M][N]) {
    float diff = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }
            diff = abs(C[i][j] - sum);
        }
    }
    return diff;
}

int main() {
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float A[M][K], B[K][N], C[M][N];
    srand(time(0));
    init(&A[0][0], M, K, 1);
    init(&B[0][0], K, N, 2);
    cout << "init C:" << endl;
    show(&C[0][0], M, N);

    cudaEventRecord(startEvent);
    MatMul(&A[0][0], &B[0][0], &C[0][0], M, N, K);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    cout << "output C:" << endl;
    show(&C[0][0], M, N);

    float diff = nativeMatMul(A, B, C);
    cout << "total err: " << diff << endl;

    printf("Average Time: %.3f ms\n", milliseconds);
}