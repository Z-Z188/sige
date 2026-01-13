#include "common.cpp"

#include <cuda.h>
#include <cuda_runtime.h>

// NOTE: keep in sync with `sige/cuda/common_cuda.cu` but extended to:
// - support float/half/bfloat16 (compute in float)
// - support more activations (relu/sigmoid/tanh)

const int threads = 512;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template<OpType opType>
__device__ __forceinline__ float binary_op_cuda(float a, float b) {
    if (opType == ADD)
        return a + b;
    else if (opType == MUL)
        return a * b;
    else
        return 0.0f;
}

template<OpType opType, typename scalar_t>
__device__ __forceinline__ float binary_op_array_cuda_4d(
        const scalar_t *__restrict__ x,
        float y,
        int B, int C, int H, int W,
        int b, int c, int h, int w
) {
    if (x == nullptr) return y;
    int p = 0;
    if (W > 1) p = w;
    if (H > 1) p += h * W;
    if (C > 1) p += c * H * W;
    if (B > 1) p += b * C * H * W;
    return binary_op_cuda<opType>(to_float(x[p]), y);
}

template<OpType opType, typename scalar_t>
__device__ __forceinline__ float binary_op_array_cuda_5d(
        const scalar_t *__restrict__ x,
        float y,
        int B, int C, int T, int H, int W,
        int b, int c, int t, int h, int w
) {
    if (x == nullptr) return y;
    int p = 0;
    if (W > 1) p = w;
    if (H > 1) p += h * W;
    if (T > 1) p += t * H * W;
    if (C > 1) p += c * T * H * W;
    if (B > 1) p += b * C * T * H * W;
    return binary_op_cuda<opType>(to_float(x[p]), y);
}

__device__ __forceinline__ float activation_cuda(ActivationType activationType, float z) {
    if (activationType == IDENTITY)
        return z;
    else if (activationType == SWISH)
        return z / (1.0f + expf(-z));
    else if (activationType == RELU)
        return z > 0.0f ? z : 0.0f;
    else if (activationType == SIGMOID)
        return 1.0f / (1.0f + expf(-z));
    else if (activationType == TANH)
        return tanhf(z);
    else
        return z;
}

