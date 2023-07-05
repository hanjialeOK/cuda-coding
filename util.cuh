#ifndef UTIL_CUH
#define UTIL_CUH

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

#endif