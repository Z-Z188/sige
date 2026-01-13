#include "common_cuda.cu"

#include <torch/extension.h>

namespace {

constexpr int rms_threads = 256;

template <typename scalar_t>
__global__ void rms_norm_channel_first_kernel(
        int64_t numVecs, int B, int C, int64_t innerSize,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ out,
        const scalar_t *__restrict__ gamma,
        const scalar_t *__restrict__ bias,
        float eps,
        float scale
) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x);
    if (vec >= numVecs) {
        return;
    }

    const int64_t bb = vec / innerSize;
    const int64_t inner = vec - bb * innerSize;
    const int64_t base0 = (bb * static_cast<int64_t>(C) * innerSize) + inner;

    extern __shared__ float shVals[];
    float sum = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        const int64_t p = base0 + static_cast<int64_t>(cc) * innerSize;
        const float v = to_float(x[p]);
        shVals[cc] = v;
        sum += v * v;
    }

    // block reduce sum
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warpSums[32];
    if (lane == 0) {
        warpSums[warp] = sum;
    }
    __syncthreads();

    __shared__ float invScale;
    float totalSum = 0.0f;
    const int numWarps = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        totalSum = (threadIdx.x < numWarps) ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            totalSum += __shfl_down_sync(0xffffffff, totalSum, offset);
        }
        if (lane == 0) {
            const float eps2 = eps * eps;
            const float denom2 = totalSum > eps2 ? totalSum : eps2;
            invScale = scale * rsqrtf(denom2);
        }
    }
    __syncthreads();

    const float inv = invScale;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        const int64_t p = base0 + static_cast<int64_t>(cc) * innerSize;
        float v = shVals[cc] * inv;
        v = v * to_float(gamma[cc]);
        if (bias != nullptr) {
            v += to_float(bias[cc]);
        }
        out[p] = from_float<scalar_t>(v);
    }
}

template <typename scalar_t>
__global__ void rms_norm_last_dim_kernel(
        int64_t numVecs, int C,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ out,
        const scalar_t *__restrict__ gamma,
        const scalar_t *__restrict__ bias,
        float eps,
        float scale
) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x);
    if (vec >= numVecs) {
        return;
    }

    const int64_t base0 = vec * static_cast<int64_t>(C);

    extern __shared__ float shVals[];
    float sum = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        const float v = to_float(x[base0 + cc]);
        shVals[cc] = v;
        sum += v * v;
    }

    // block reduce sum
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warpSums[32];
    if (lane == 0) {
        warpSums[warp] = sum;
    }
    __syncthreads();

    __shared__ float invScale;
    float totalSum = 0.0f;
    const int numWarps = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        totalSum = (threadIdx.x < numWarps) ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            totalSum += __shfl_down_sync(0xffffffff, totalSum, offset);
        }
        if (lane == 0) {
            const float eps2 = eps * eps;
            const float denom2 = totalSum > eps2 ? totalSum : eps2;
            invScale = scale * rsqrtf(denom2);
        }
    }
    __syncthreads();

    const float inv = invScale;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float v = shVals[cc] * inv;
        v = v * to_float(gamma[cc]);
        if (bias != nullptr) {
            v += to_float(bias[cc]);
        }
        out[base0 + cc] = from_float<scalar_t>(v);
    }
}

}  // namespace

torch::Tensor rms_norm_cuda(
        const torch::Tensor &x,
        const torch::Tensor &gamma,
        const torch::optional<torch::Tensor> &bias,
        double eps,
        bool channelFirst
) {
    TORCH_CHECK(x.is_cuda(), "rms_norm_cuda: x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "rms_norm_cuda: x must be contiguous");
    TORCH_CHECK(x.dim() >= 2, "rms_norm_cuda: x must have at least 2 dims");

    TORCH_CHECK(gamma.is_cuda(), "rms_norm_cuda: gamma must be CUDA");
    TORCH_CHECK(gamma.is_contiguous(), "rms_norm_cuda: gamma must be contiguous");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "rms_norm_cuda: gamma dtype must match x");

    const int C = channelFirst ? static_cast<int>(x.size(1)) : static_cast<int>(x.size(x.dim() - 1));
    TORCH_CHECK(gamma.numel() == C, "rms_norm_cuda: gamma.numel() must equal normalized dim");

    const float epsF = static_cast<float>(eps);
    const float scale = sqrtf(static_cast<float>(C));

    auto out = torch::empty_like(x);
    if (x.numel() == 0) {
        return out;
    }

    const int64_t numVecs = x.numel() / C;

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "rms_norm_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outData = out.data_ptr<scalar_t>();
                const scalar_t *gammaData = gamma.data_ptr<scalar_t>();

                const scalar_t *biasData = nullptr;
                if (bias.has_value()) {
                    TORCH_CHECK(bias.value().is_cuda(), "rms_norm_cuda: bias must be CUDA");
                    TORCH_CHECK(bias.value().is_contiguous(), "rms_norm_cuda: bias must be contiguous");
                    TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "rms_norm_cuda: bias dtype must match x");
                    TORCH_CHECK(bias.value().numel() == C, "rms_norm_cuda: bias.numel() must equal normalized dim");
                    biasData = bias.value().data_ptr<scalar_t>();
                }

                const dim3 blocks(static_cast<unsigned int>(numVecs), 1, 1);
                const int normThreads = (C <= 32) ? 32 : (C <= 64) ? 64 : (C <= 128) ? 128 : rms_threads;
                const size_t shmemBytes = static_cast<size_t>(C) * sizeof(float);
                if (channelFirst) {
                    const int B = static_cast<int>(x.size(0));
                    const int64_t innerSize = x.numel() / (static_cast<int64_t>(B) * C);
                    rms_norm_channel_first_kernel<scalar_t><<<blocks, normThreads, shmemBytes>>>(
                            numVecs, B, C, innerSize,
                            xData, outData,
                            gammaData, biasData,
                            epsF, scale);
                } else {
                    rms_norm_last_dim_kernel<scalar_t><<<blocks, normThreads, shmemBytes>>>(
                            numVecs, C,
                            xData, outData,
                            gammaData, biasData,
                            epsF, scale);
                }
            });

    return out;
}
