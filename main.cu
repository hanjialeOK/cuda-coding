#include<iostream>
#include<vector>
#include<cuda.h>
#include<cuda_runtime.h>

#include "util.cuh"

using namespace std;

const unsigned M = 128;
const unsigned N = 128;
const unsigned K = 128;

void gemmBasic(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
            unsigned M, unsigned N, unsigned K);
void gemmUse128(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
            unsigned M, unsigned N, unsigned K);
void gemmUseTile(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
            unsigned M, unsigned N, unsigned K);

void init(float *A, int w, int h, int val=1) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            A[i*w+j] = rand() % 5;
        }
    }
}

void show(const float *A, int w, int h, int size=10) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << A[i*w+j] << " ";
        }
        cout << endl;
    }
}

void computeANS(const float *A, const float *B, float *ANS) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K+k] * B[k*N+j];
            }
            ANS[i*N+j] = sum;
        }
    }
}

float computeDiff(const float *C, const float *ANS) {
    float diff = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            diff += abs(ANS[i*N+j] - C[i*N+j]);
        }
    }
    return diff;
}

void evaluate(const float *A, const float *B, float *C, const float *ANS, string algo) {
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    if (algo == "basic") {
        gemmBasic(A, B, C, M, N, K);
    } else if (algo == "use_128") {
        gemmUse128(A, B, C, M, N, K);
    } else if (algo == "use_tile") {
        gemmUseTile(A, B, C, M, N, K);
    } else {
        cout << "Unkonwn algo!" << endl;
    }
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    float diff = computeDiff(C, ANS);

    cout << "algo: " << algo << ", total err: " << diff << ", duration: " << milliseconds << endl;
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

    float *A, *B, *C, *ANS;
    A = (float*)malloc(M*K*sizeof(float));
    B = (float*)malloc(K*N*sizeof(float));
    C = (float*)malloc(M*N*sizeof(float));
    ANS = (float*)malloc(M*N*sizeof(float));
    srand(time(0));
    init(A, M, K, 1);
    init(B, K, N, 2);
    cout << "init C:" << endl;
    show(C, M, N);

    cout << "computing ANS:" << endl;
    computeANS(A, B, ANS);
    cout << "ANS done:" << endl;

    vector<float> duration;
    vector<float> diff;

    evaluate(A, B, C, ANS, "basic");
    cout << "basic output C:" << endl;
    show(C, M, N);
    evaluate(A, B, C, ANS, "use_128");
    cout << "use_128 output C:" << endl;
    show(C, M, N);
    evaluate(A, B, C, ANS, "use_tile");
    cout << "use_tile output C:" << endl;
    show(C, M, N);

    free(A);
    free(B);
    free(C);
}