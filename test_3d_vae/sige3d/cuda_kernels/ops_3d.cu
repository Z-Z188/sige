#include "common_cuda.cu"

#include <limits>
#include <torch/extension.h>

template <typename scalar_t, typename index_t>
__global__ void gather3d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int T, int H, int W,
        int R, int S,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleT, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftT, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int tt = static_cast<int>(t % T);
    t /= T;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh;
    if (hh < 0 || hh >= H) {
        output[index] = from_float<scalar_t>(0.0f);
        return;
    }
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww < 0 || ww >= W) {
        output[index] = from_float<scalar_t>(0.0f);
        return;
    }

    int64_t p = ((static_cast<int64_t>(bb) * C + cc) * T + tt) * H * W + static_cast<int64_t>(hh) * W + ww;
    float z = to_float(x[p]);
    if (!activationFirst) {
        z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
        z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
    }
    z = activation_cuda(activationType, z);
    if (activationFirst) {
        z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
        z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
    }
    output[index] = from_float<scalar_t>(z);
}

torch::Tensor gather3d_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    TORCH_CHECK(x.is_cuda(), "gather3d_cuda: x must be CUDA");
    TORCH_CHECK(x.dim() == 5, "gather3d_cuda: x must be 5D [B,C,T,H,W]");
    TORCH_CHECK(activeIndices.is_cuda(), "gather3d_cuda: activeIndices must be CUDA");
    TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "gather3d_cuda: activeIndices must be int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "gather3d_cuda: activeIndices must be [N,2]");

    const int R = bSizeH, S = bSizeW;
    const int numActive = static_cast<int>(activeIndices.size(0));
    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int H = static_cast<int>(x.size(3));
    const int W = static_cast<int>(x.size(4));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, T, R, S}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const auto activationType = getActivationType(activationName);
    const int *activeIndicesData = activeIndices.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "gather3d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleT = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "gather3d_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "gather3d_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 5, "gather3d_cuda: scale must be 5D");
                    TORCH_CHECK(broadcastable(x, scale.value()), "gather3d_cuda: scale not broadcastable to x");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleT = static_cast<int>(scale.value().size(2));
                    scaleH = static_cast<int>(scale.value().size(3));
                    scaleW = static_cast<int>(scale.value().size(4));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftT = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "gather3d_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "gather3d_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 5, "gather3d_cuda: shift must be 5D");
                    TORCH_CHECK(broadcastable(x, shift.value()), "gather3d_cuda: shift not broadcastable to x");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftT = static_cast<int>(shift.value().size(2));
                    shiftH = static_cast<int>(shift.value().size(3));
                    shiftW = static_cast<int>(shift.value().size(4));
                }

                const int64_t total = output.numel();
                const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks32(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    gather3d_cuda_kernel<scalar_t, int><<<blocks32, threads>>>(
                            total32, numActive,
                            B, C, T, H, W,
                            R, S,
                            xData, outputData, activeIndicesData,
                            scaleData,
                            scaleB, scaleC, scaleT, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftT, shiftH, shiftW,
                            activationType, activationFirst);
                } else {
                    gather3d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, T, H, W,
                            R, S,
                            xData, outputData, activeIndicesData,
                            scaleData,
                            scaleB, scaleC, scaleT, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftT, shiftH, shiftW,
                            activationType, activationFirst);
                }
            });

    return output;
}


namespace {

constexpr int rms_norm_threads = 256;

template <typename scalar_t>
__global__ void gather3d_rmsnorm_cuda_kernel(
        int64_t numVecs, int numActive,
        int B, int C, int T, int H, int W,
        int R, int S,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const scalar_t *__restrict__ gamma,
        const scalar_t *__restrict__ bias,
        float eps,
        float rmsScale,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleT, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftT, int shiftH, int shiftW,
        ActivationType activationType) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x);
    if (vec >= numVecs) {
        return;
    }

    int64_t t = vec;
    const int intraBw = static_cast<int>(t % S);
    t /= S;
    const int intraBh = static_cast<int>(t % R);
    t /= R;
    const int tt = static_cast<int>(t % T);
    t /= T;
    const int ib = static_cast<int>(t % numActive);
    const int bb = static_cast<int>(t / numActive);

    const int biH = activeIndices[ib << 1];
    const int hh = biH + intraBh;
    const int biW = activeIndices[ib << 1 | 1];
    const int ww = biW + intraBw;

    const int64_t inStrideC = static_cast<int64_t>(T) * H * W;
    const int64_t outStrideC = static_cast<int64_t>(T) * R * S;
    const int64_t outBaseVec = (static_cast<int64_t>(bb) * numActive + ib) * static_cast<int64_t>(C) * outStrideC +
                               static_cast<int64_t>(tt) * R * S + intraBh * S + intraBw;

    // invalid spatial => output must be all zeros (keep consistent even for sigmoid).
    if (hh < 0 || hh >= H || ww < 0 || ww >= W) {
        for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
            output[outBaseVec + static_cast<int64_t>(cc) * outStrideC] = from_float<scalar_t>(0.0f);
        }
        return;
    }

    const int64_t inBase = (static_cast<int64_t>(bb) * C) * inStrideC + static_cast<int64_t>(tt) * H * W + hh * W + ww;

    extern __shared__ float shVals[];
    float sum = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        const float v = to_float(x[inBase + static_cast<int64_t>(cc) * inStrideC]);
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
            invScale = rmsScale * rsqrtf(denom2);
        }
    }
    __syncthreads();

    const float inv = invScale;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float z = shVals[cc] * inv;
        z = z * to_float(gamma[cc]);
        if (bias != nullptr) {
            z += to_float(bias[cc]);
        }

        z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
        z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
        z = activation_cuda(activationType, z);

        output[outBaseVec + static_cast<int64_t>(cc) * outStrideC] = from_float<scalar_t>(z);
    }
}

}  // namespace

torch::Tensor gather3d_rmsnorm_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &gamma,
        const torch::optional<torch::Tensor> &bias,
        double eps,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity")) {
    TORCH_CHECK(x.is_cuda(), "gather3d_rmsnorm_cuda: x must be CUDA");
    TORCH_CHECK(x.dim() == 5, "gather3d_rmsnorm_cuda: x must be 5D [B,C,T,H,W]");
    TORCH_CHECK(activeIndices.is_cuda(), "gather3d_rmsnorm_cuda: activeIndices must be CUDA");
    TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "gather3d_rmsnorm_cuda: activeIndices must be int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "gather3d_rmsnorm_cuda: activeIndices must be [N,2]");

    TORCH_CHECK(gamma.is_cuda(), "gather3d_rmsnorm_cuda: gamma must be CUDA");
    TORCH_CHECK(gamma.is_contiguous(), "gather3d_rmsnorm_cuda: gamma must be contiguous");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "gather3d_rmsnorm_cuda: gamma dtype must match x");

    const int R = bSizeH, S = bSizeW;
    const int numActive = static_cast<int>(activeIndices.size(0));
    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int H = static_cast<int>(x.size(3));
    const int W = static_cast<int>(x.size(4));

    TORCH_CHECK(gamma.numel() == C, "gather3d_rmsnorm_cuda: gamma.numel() must equal C");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, T, R, S}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const auto activationType = getActivationType(activationName);
    const int *activeIndicesData = activeIndices.data_ptr<int>();
    const float epsF = static_cast<float>(eps);
    const float rmsScale = sqrtf(static_cast<float>(C));

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "gather3d_rmsnorm_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *gammaData = gamma.data_ptr<scalar_t>();
                const scalar_t *biasData = nullptr;
                if (bias.has_value()) {
                    TORCH_CHECK(bias.value().is_cuda(), "gather3d_rmsnorm_cuda: bias must be CUDA");
                    TORCH_CHECK(bias.value().is_contiguous(), "gather3d_rmsnorm_cuda: bias must be contiguous");
                    TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "gather3d_rmsnorm_cuda: bias dtype must match x");
                    TORCH_CHECK(bias.value().numel() == C, "gather3d_rmsnorm_cuda: bias.numel() must equal C");
                    biasData = bias.value().data_ptr<scalar_t>();
                }

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleT = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "gather3d_rmsnorm_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "gather3d_rmsnorm_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 5, "gather3d_rmsnorm_cuda: scale must be 5D");
                    TORCH_CHECK(broadcastable(x, scale.value()), "gather3d_rmsnorm_cuda: scale not broadcastable to x");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleT = static_cast<int>(scale.value().size(2));
                    scaleH = static_cast<int>(scale.value().size(3));
                    scaleW = static_cast<int>(scale.value().size(4));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftT = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "gather3d_rmsnorm_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "gather3d_rmsnorm_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 5, "gather3d_rmsnorm_cuda: shift must be 5D");
                    TORCH_CHECK(broadcastable(x, shift.value()), "gather3d_rmsnorm_cuda: shift not broadcastable to x");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftT = static_cast<int>(shift.value().size(2));
                    shiftH = static_cast<int>(shift.value().size(3));
                    shiftW = static_cast<int>(shift.value().size(4));
                }

                const int64_t numVecs = static_cast<int64_t>(B) * numActive * T * R * S;
                const dim3 blocks(static_cast<unsigned int>(numVecs), 1, 1);

                const int normThreads = (C <= 32) ? 32 : (C <= 64) ? 64 : (C <= 128) ? 128 : rms_norm_threads;
                const size_t shmemBytes = static_cast<size_t>(C) * sizeof(float);

                gather3d_rmsnorm_cuda_kernel<scalar_t><<<blocks, normThreads, shmemBytes>>>(
                        numVecs, numActive,
                        B, C, T, H, W,
                        R, S,
                        xData, outputData,
                        activeIndicesData,
                        gammaData,
                        biasData,
                        epsF,
                        rmsScale,
                        scaleData,
                        scaleB, scaleC, scaleT, scaleH, scaleW,
                        shiftData,
                        shiftB, shiftC, shiftT, shiftH, shiftW,
                        activationType);
            });

    return output;
}

template <typename scalar_t, typename index_t, bool residualIsFull, bool outputIndex32>
__global__ void scatter3d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int T, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        int strideHShift, int strideWShift,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const scalar_t *__restrict__ residual,
        int residualB, int residualC, int residualT, int residualH, int residualW) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int tt = static_cast<int>(t % T);
    t /= T;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = 0, biW = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned int mask = __match_any_sync(0xffffffffu, ib);
    const int leader = __ffs(mask) - 1;

    int biHTmp = 0, biWTmp = 0;
    if (lane == leader) {
        const int aiH = activeIndices[ib << 1];
        const int aiW = activeIndices[ib << 1 | 1];
        const int offH = offsetH + aiH;
        const int offW = offsetW + aiW;
        biHTmp = (strideHShift >= 0) ? (offH >> strideHShift) : (offH / strideH);
        biWTmp = (strideWShift >= 0) ? (offW >> strideWShift) : (offW / strideW);
    }
    biH = __shfl_sync(mask, biHTmp, leader);
    biW = __shfl_sync(mask, biWTmp, leader);
#else
    biH = (strideHShift >= 0) ? ((offsetH + activeIndices[ib << 1]) >> strideHShift)
                              : ((offsetH + activeIndices[ib << 1]) / strideH);
    biW = (strideWShift >= 0) ? ((offsetW + activeIndices[ib << 1 | 1]) >> strideWShift)
                              : ((offsetW + activeIndices[ib << 1 | 1]) / strideW);
#endif

    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int ww = biW + intraBw;
    if (ww >= W)
        return;

    float z = to_float(x[index]);
    if (outputIndex32) {
        const int p = ((bb * C + cc) * T + tt) * H * W + hh * W + ww;
        if (residualIsFull) {
            z += to_float(residual[p]);
        } else {
            z = binary_op_array_cuda_5d<ADD>(residual, z, residualB, residualC, residualT, residualH, residualW, bb, cc, tt, hh, ww);
        }
        output[p] = from_float<scalar_t>(z);
    } else {
        const int64_t p = ((static_cast<int64_t>(bb) * C + cc) * T + tt) * H * W + static_cast<int64_t>(hh) * W + ww;
        if (residualIsFull) {
            z += to_float(residual[p]);
        } else {
            z = binary_op_array_cuda_5d<ADD>(residual, z, residualB, residualC, residualT, residualH, residualW, bb, cc, tt, hh, ww);
        }
        output[p] = from_float<scalar_t>(z);
    }
}

template <typename scalar_t, typename index_t, bool outputIndex32>
__global__ void calibrate_residual3d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int T, int H, int W,
        int R, int S,
        const scalar_t *__restrict__ x,
        const scalar_t *__restrict__ y,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int tt = static_cast<int>(t % T);
    t /= T;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = 0, biW = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned int mask = __match_any_sync(0xffffffffu, ib);
    const int leader = __ffs(mask) - 1;

    int biHTmp = 0, biWTmp = 0;
    if (lane == leader) {
        biHTmp = activeIndices[ib << 1];
        biWTmp = activeIndices[ib << 1 | 1];
    }
    biH = __shfl_sync(mask, biHTmp, leader);
    biW = __shfl_sync(mask, biWTmp, leader);
#else
    biH = activeIndices[ib << 1];
    biW = activeIndices[ib << 1 | 1];
#endif

    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int ww = biW + intraBw;
    if (ww >= W)
        return;

    if (outputIndex32) {
        const int p = ((bb * C + cc) * T + tt) * H * W + hh * W + ww;
        float cur = to_float(output[p]);
        cur += to_float(x[index]) - to_float(y[p]);
        output[p] = from_float<scalar_t>(cur);
    } else {
        const int64_t p = ((static_cast<int64_t>(bb) * C + cc) * T + tt) * H * W + static_cast<int64_t>(hh) * W + ww;
        float cur = to_float(output[p]);
        cur += to_float(x[index]) - to_float(y[p]);
        output[p] = from_float<scalar_t>(cur);
    }
}

torch::Tensor scatter3d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &residual) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "scatter3d_cuda: x/y must be CUDA");
    TORCH_CHECK(x.dim() == 5, "scatter3d_cuda: x must be 5D [B*numActive,C,T,R,S]");
    TORCH_CHECK(y.dim() == 5, "scatter3d_cuda: y must be 5D [B,C,T,H,W]");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "scatter3d_cuda: x/y dtype must match");
    TORCH_CHECK(activeIndices.is_cuda() && activeIndices.scalar_type() == torch::kInt32, "scatter3d_cuda: activeIndices must be CUDA int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "scatter3d_cuda: activeIndices must be [N,2]");

    const int numActive = static_cast<int>(activeIndices.size(0));
    auto output = y.clone();
    if (numActive == 0 || x.numel() == 0) {
        return output;
    }

    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int T = static_cast<int>(y.size(2));
    const int H = static_cast<int>(y.size(3));
    const int W = static_cast<int>(y.size(4));

    TORCH_CHECK(x.size(0) == B * numActive, "scatter3d_cuda: x.size(0) must equal B*numActive");
    TORCH_CHECK(x.size(1) == C, "scatter3d_cuda: x.size(1) must equal C");
    TORCH_CHECK(x.size(2) == T, "scatter3d_cuda: x.size(2) must equal T");

    const int R = static_cast<int>(x.size(3));
    const int S = static_cast<int>(x.size(4));

    const int *activeIndicesData = activeIndices.data_ptr<int>();
    const bool outputIndex32 = (y.numel() <= static_cast<int64_t>(std::numeric_limits<int>::max()));

    auto stride_shift_or_neg1 = [](int stride) -> int {
        if (stride <= 0) return -1;
        const unsigned int u = static_cast<unsigned int>(stride);
        if ((u & (u - 1)) != 0) return -1;
        return __builtin_ctz(u);
    };
    const int strideHShift = stride_shift_or_neg1(strideH);
    const int strideWShift = stride_shift_or_neg1(strideW);

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "scatter3d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *residualData = nullptr;
                int residualB = 0, residualC = 0, residualT = 0, residualH = 0, residualW = 0;
                bool residualIsFull = false;
                if (residual.has_value()) {
                    TORCH_CHECK(residual.value().is_cuda(), "scatter3d_cuda: residual must be CUDA");
                    TORCH_CHECK(residual.value().scalar_type() == x.scalar_type(), "scatter3d_cuda: residual dtype must match x");
                    TORCH_CHECK(residual.value().dim() == 5, "scatter3d_cuda: residual must be 5D");
                    TORCH_CHECK(broadcastable(y, residual.value()), "scatter3d_cuda: residual not broadcastable to y");
                    residualData = residual.value().data_ptr<scalar_t>();
                    residualB = static_cast<int>(residual.value().size(0));
                    residualC = static_cast<int>(residual.value().size(1));
                    residualT = static_cast<int>(residual.value().size(2));
                    residualH = static_cast<int>(residual.value().size(3));
                    residualW = static_cast<int>(residual.value().size(4));
                    residualIsFull = (residualB == B) && (residualC == C) && (residualT == T) && (residualH == H) && (residualW == W);
                }

                const int64_t total = x.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    if (outputIndex32) {
                        if (residualIsFull) {
                            scatter3d_cuda_kernel<scalar_t, int, true, true><<<blocks, threads>>>(
                                    total32, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        } else {
                            scatter3d_cuda_kernel<scalar_t, int, false, true><<<blocks, threads>>>(
                                    total32, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        }
                    } else {
                        if (residualIsFull) {
                            scatter3d_cuda_kernel<scalar_t, int, true, false><<<blocks, threads>>>(
                                    total32, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        } else {
                            scatter3d_cuda_kernel<scalar_t, int, false, false><<<blocks, threads>>>(
                                    total32, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        }
                    }
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    if (outputIndex32) {
                        if (residualIsFull) {
                            scatter3d_cuda_kernel<scalar_t, int64_t, true, true><<<blocks, threads>>>(
                                    total, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        } else {
                            scatter3d_cuda_kernel<scalar_t, int64_t, false, true><<<blocks, threads>>>(
                                    total, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        }
                    } else {
                        if (residualIsFull) {
                            scatter3d_cuda_kernel<scalar_t, int64_t, true, false><<<blocks, threads>>>(
                                    total, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        } else {
                            scatter3d_cuda_kernel<scalar_t, int64_t, false, false><<<blocks, threads>>>(
                                    total, numActive,
                                    B, C, T, H, W,
                                    R, S,
                                    offsetH, offsetW,
                                    strideH, strideW,
                                    strideHShift, strideWShift,
                                    xData, outputData,
                                    activeIndicesData,
                                    residualData,
                                    residualB, residualC, residualT, residualH, residualW);
                        }
                    }
                }
            });

    return output;
}

torch::Tensor scatter_with_block_residual3d_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1) {
    auto output = scatter3d_cuda(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);
    if (x1.numel() == 0 || activeIndices1.numel() == 0) {
        return output;
    }

    TORCH_CHECK(x1.is_cuda() && y1.is_cuda(), "scatter_with_block_residual3d_cuda: x1/y1 must be CUDA");
    TORCH_CHECK(x1.scalar_type() == y1.scalar_type(), "scatter_with_block_residual3d_cuda: x1/y1 dtype must match");
    TORCH_CHECK(x1.scalar_type() == output.scalar_type(), "scatter_with_block_residual3d_cuda: dtype mismatch");
    TORCH_CHECK(activeIndices1.is_cuda() && activeIndices1.scalar_type() == torch::kInt32, "scatter_with_block_residual3d_cuda: activeIndices1 must be CUDA int32");

    const int B = static_cast<int>(y1.size(0));
    const int C = static_cast<int>(y1.size(1));
    const int T = static_cast<int>(y1.size(2));
    const int H = static_cast<int>(y1.size(3));
    const int W = static_cast<int>(y1.size(4));

    const int numActive = static_cast<int>(activeIndices1.size(0));
    TORCH_CHECK(x1.size(0) == B * numActive, "scatter_with_block_residual3d_cuda: x1.size(0) must equal B*numActive1");
    TORCH_CHECK(x1.size(1) == C, "scatter_with_block_residual3d_cuda: x1.size(1) must equal C");
    TORCH_CHECK(x1.size(2) == T, "scatter_with_block_residual3d_cuda: x1.size(2) must equal T");

    const int R = static_cast<int>(x1.size(3));
    const int S = static_cast<int>(x1.size(4));

    const int *activeIndicesData = activeIndices1.data_ptr<int>();
    const bool outputIndex32 = (y1.numel() <= static_cast<int64_t>(std::numeric_limits<int>::max()));

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x1.scalar_type(),
            "scatter_with_block_residual3d_cuda",
            [&] {
                const scalar_t *x1Data = x1.data_ptr<scalar_t>();
                const scalar_t *y1Data = y1.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const int64_t total = x1.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    if (outputIndex32) {
                        calibrate_residual3d_cuda_kernel<scalar_t, int, true><<<blocks, threads>>>(
                                total32, numActive,
                                B, C, T, H, W,
                                R, S,
                                x1Data, y1Data,
                                outputData,
                                activeIndicesData);
                    } else {
                        calibrate_residual3d_cuda_kernel<scalar_t, int, false><<<blocks, threads>>>(
                                total32, numActive,
                                B, C, T, H, W,
                                R, S,
                                x1Data, y1Data,
                                outputData,
                                activeIndicesData);
                    }
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    if (outputIndex32) {
                        calibrate_residual3d_cuda_kernel<scalar_t, int64_t, true><<<blocks, threads>>>(
                                total, numActive,
                                B, C, T, H, W,
                                R, S,
                                x1Data, y1Data,
                                outputData,
                                activeIndicesData);
                    } else {
                        calibrate_residual3d_cuda_kernel<scalar_t, int64_t, false><<<blocks, threads>>>(
                                total, numActive,
                                B, C, T, H, W,
                                R, S,
                                x1Data, y1Data,
                                outputData,
                                activeIndicesData);
                    }
                }
            });

    return output;
}

template <typename scalar_t, typename index_t>
__global__ void scatter_gather3d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int T, int H, int W,
        int Rx, int Sx,
        int Ro, int So,
        const scalar_t *__restrict__ x,
        const scalar_t *__restrict__ y,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const int *__restrict__ scatterMap,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleT, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftT, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % So);
    t /= So;
    int intraBh = static_cast<int>(t % Ro);
    t /= Ro;
    int tt = static_cast<int>(t % T);
    t /= T;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh;
    if (hh < 0 || hh >= H) {
        output[index] = from_float<scalar_t>(0.0f);
        return;
    }
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww < 0 || ww >= W) {
        output[index] = from_float<scalar_t>(0.0f);
        return;
    }

    int scatterMapIndex = (hh * W + ww) * 3;
    int bx = scatterMap[scatterMapIndex];

    int64_t py = ((static_cast<int64_t>(bb) * C + cc) * T + tt) * H * W + static_cast<int64_t>(hh) * W + ww;

    float z;
    if (bx >= 0) {
        int hx = scatterMap[scatterMapIndex + 1];
        int wx = scatterMap[scatterMapIndex + 2];
        int64_t px = ((((static_cast<int64_t>(bb) * numActive + bx) * C + cc) * T + tt) * Rx + hx) * Sx + wx;
        z = to_float(x[px]);
    } else {
        z = to_float(y[py]);
    }

    if (!activationFirst) {
        z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
        z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
    }
    z = activation_cuda(activationType, z);
    if (activationFirst) {
        z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
        z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
    }

    output[index] = from_float<scalar_t>(z);
}

torch::Tensor scatter_gather3d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "scatter_gather3d_cuda: x/y must be CUDA");
    TORCH_CHECK(x.dim() == 5 && y.dim() == 5, "scatter_gather3d_cuda: x/y must be 5D");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "scatter_gather3d_cuda: x/y dtype must match");
    TORCH_CHECK(activeIndices.is_cuda() && activeIndices.scalar_type() == torch::kInt32, "scatter_gather3d_cuda: activeIndices must be CUDA int32");
    TORCH_CHECK(scatterMap.is_cuda() && scatterMap.scalar_type() == torch::kInt32, "scatter_gather3d_cuda: scatterMap must be CUDA int32");

    const int Ro = bSizeH, So = bSizeW;
    const int Rx = static_cast<int>(x.size(3));
    const int Sx = static_cast<int>(x.size(4));
    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int T = static_cast<int>(y.size(2));
    const int H = static_cast<int>(y.size(3));
    const int W = static_cast<int>(y.size(4));

    const int numActive = static_cast<int>(activeIndices.size(0));
    TORCH_CHECK(x.size(1) == C, "scatter_gather3d_cuda: x.size(1) must equal C");
    TORCH_CHECK(x.size(2) == T, "scatter_gather3d_cuda: x.size(2) must equal T");
    TORCH_CHECK(x.size(0) == B * numActive, "scatter_gather3d_cuda: x.size(0) must equal B*numActive");
    TORCH_CHECK(scatterMap.size(0) == H && scatterMap.size(1) == W && scatterMap.size(2) == 3, "scatter_gather3d_cuda: scatterMap must be [H,W,3]");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, T, Ro, So}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const auto activationType = getActivationType(activationName);

    const int *activeIndicesData = activeIndices.data_ptr<int>();
    const int *scatterMapData = scatterMap.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "scatter_gather3d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                const scalar_t *yData = y.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleT = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "scatter_gather3d_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "scatter_gather3d_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 5, "scatter_gather3d_cuda: scale must be 5D");
                    TORCH_CHECK(broadcastable(y, scale.value()), "scatter_gather3d_cuda: scale not broadcastable to y");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleT = static_cast<int>(scale.value().size(2));
                    scaleH = static_cast<int>(scale.value().size(3));
                    scaleW = static_cast<int>(scale.value().size(4));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftT = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "scatter_gather3d_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "scatter_gather3d_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 5, "scatter_gather3d_cuda: shift must be 5D");
                    TORCH_CHECK(broadcastable(y, shift.value()), "scatter_gather3d_cuda: shift not broadcastable to y");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftT = static_cast<int>(shift.value().size(2));
                    shiftH = static_cast<int>(shift.value().size(3));
                    shiftW = static_cast<int>(shift.value().size(4));
                }

                const int64_t total = output.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    scatter_gather3d_cuda_kernel<scalar_t, int><<<blocks, threads>>>(
                            total32, numActive,
                            B, C, T, H, W,
                            Rx, Sx,
                            Ro, So,
                            xData, yData, outputData,
                            activeIndicesData, scatterMapData,
                            scaleData,
                            scaleB, scaleC, scaleT, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftT, shiftH, shiftW,
                            activationType, activationFirst);
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    scatter_gather3d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, T, H, W,
                            Rx, Sx,
                            Ro, So,
                            xData, yData, outputData,
                            activeIndicesData, scatterMapData,
                            scaleData,
                            scaleB, scaleC, scaleT, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftT, shiftH, shiftW,
                            activationType, activationFirst);
                }
            });

    return output;
}

namespace {

template <typename scalar_t>
__global__ void scatter_gather3d_rmsnorm_cuda_kernel(
        int64_t numVecs, int numActive,
        int B, int C, int T, int H, int W,
        int Rx, int Sx,
        int Ro, int So,
        const scalar_t *__restrict__ x,
        const scalar_t *__restrict__ y,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const int *__restrict__ scatterMap,
        const scalar_t *__restrict__ gamma,
        const scalar_t *__restrict__ bias,
        float eps,
        float rmsScale,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleT, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftT, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst) {
    const int64_t vec = static_cast<int64_t>(blockIdx.x);
    if (vec >= numVecs) {
        return;
    }

    int64_t t = vec;
    const int intraBw = static_cast<int>(t % So);
    t /= So;
    const int intraBh = static_cast<int>(t % Ro);
    t /= Ro;
    const int tt = static_cast<int>(t % T);
    t /= T;
    const int ib = static_cast<int>(t % numActive);
    const int bb = static_cast<int>(t / numActive);

    const int biH = activeIndices[ib << 1];
    const int hh = biH + intraBh;
    const int biW = activeIndices[ib << 1 | 1];
    const int ww = biW + intraBw;

    const int64_t outStrideC = static_cast<int64_t>(T) * Ro * So;
    const int64_t outBaseVec = (static_cast<int64_t>(bb) * numActive + ib) * static_cast<int64_t>(C) * outStrideC +
                               static_cast<int64_t>(tt) * Ro * So + intraBh * So + intraBw;

    if (hh < 0 || hh >= H || ww < 0 || ww >= W) {
        for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
            output[outBaseVec + static_cast<int64_t>(cc) * outStrideC] = from_float<scalar_t>(0.0f);
        }
        return;
    }

    const int scatterMapIndex = (hh * W + ww) * 3;
    const int bx = scatterMap[scatterMapIndex];

    const int64_t yStrideC = static_cast<int64_t>(T) * H * W;
    const int64_t yBase = (static_cast<int64_t>(bb) * C) * yStrideC + static_cast<int64_t>(tt) * H * W + hh * W + ww;

    int hx = 0, wx = 0;
    int64_t xBase = 0;
    const int64_t xStrideC = static_cast<int64_t>(T) * Rx * Sx;
    if (bx >= 0) {
        hx = scatterMap[scatterMapIndex + 1];
        wx = scatterMap[scatterMapIndex + 2];
        // same indexing as scatter_gather3d_cuda_kernel
        xBase = (((static_cast<int64_t>(bb) * numActive + bx) * C) * xStrideC) +
                static_cast<int64_t>(tt) * Rx * Sx + hx * Sx + wx;
    }

    extern __shared__ float shVals[];
    float sum = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float v;
        if (bx >= 0) {
            v = to_float(x[xBase + static_cast<int64_t>(cc) * xStrideC]);
        } else {
            v = to_float(y[yBase + static_cast<int64_t>(cc) * yStrideC]);
        }
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
            invScale = rmsScale * rsqrtf(denom2);
        }
    }
    __syncthreads();

    const float inv = invScale;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float z = shVals[cc] * inv;
        z = z * to_float(gamma[cc]);
        if (bias != nullptr) {
            z += to_float(bias[cc]);
        }

        if (!activationFirst) {
            z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
            z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
        }
        z = activation_cuda(activationType, z);
        if (activationFirst) {
            z = binary_op_array_cuda_5d<MUL>(scale, z, scaleB, scaleC, scaleT, scaleH, scaleW, bb, cc, tt, hh, ww);
            z = binary_op_array_cuda_5d<ADD>(shift, z, shiftB, shiftC, shiftT, shiftH, shiftW, bb, cc, tt, hh, ww);
        }

        output[outBaseVec + static_cast<int64_t>(cc) * outStrideC] = from_float<scalar_t>(z);
    }
}

}  // namespace

torch::Tensor scatter_gather3d_rmsnorm_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &scatterMap,
        const torch::Tensor &gamma,
        const torch::optional<torch::Tensor> &bias,
        double eps,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "scatter_gather3d_rmsnorm_cuda: x/y must be CUDA");
    TORCH_CHECK(x.dim() == 5 && y.dim() == 5, "scatter_gather3d_rmsnorm_cuda: x/y must be 5D");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "scatter_gather3d_rmsnorm_cuda: x/y dtype must match");
    TORCH_CHECK(activeIndices.is_cuda() && activeIndices.scalar_type() == torch::kInt32, "scatter_gather3d_rmsnorm_cuda: activeIndices must be CUDA int32");
    TORCH_CHECK(scatterMap.is_cuda() && scatterMap.scalar_type() == torch::kInt32, "scatter_gather3d_rmsnorm_cuda: scatterMap must be CUDA int32");

    TORCH_CHECK(gamma.is_cuda(), "scatter_gather3d_rmsnorm_cuda: gamma must be CUDA");
    TORCH_CHECK(gamma.is_contiguous(), "scatter_gather3d_rmsnorm_cuda: gamma must be contiguous");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "scatter_gather3d_rmsnorm_cuda: gamma dtype must match x");

    const int Ro = bSizeH, So = bSizeW;
    const int Rx = static_cast<int>(x.size(3));
    const int Sx = static_cast<int>(x.size(4));
    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int T = static_cast<int>(y.size(2));
    const int H = static_cast<int>(y.size(3));
    const int W = static_cast<int>(y.size(4));

    const int numActive = static_cast<int>(activeIndices.size(0));
    TORCH_CHECK(x.size(1) == C, "scatter_gather3d_rmsnorm_cuda: x.size(1) must equal C");
    TORCH_CHECK(x.size(2) == T, "scatter_gather3d_rmsnorm_cuda: x.size(2) must equal T");
    TORCH_CHECK(x.size(0) == B * numActive, "scatter_gather3d_rmsnorm_cuda: x.size(0) must equal B*numActive");
    TORCH_CHECK(scatterMap.size(0) == H && scatterMap.size(1) == W && scatterMap.size(2) == 3, "scatter_gather3d_rmsnorm_cuda: scatterMap must be [H,W,3]");
    TORCH_CHECK(gamma.numel() == C, "scatter_gather3d_rmsnorm_cuda: gamma.numel() must equal C");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, T, Ro, So}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const auto activationType = getActivationType(activationName);
    const int *activeIndicesData = activeIndices.data_ptr<int>();
    const int *scatterMapData = scatterMap.data_ptr<int>();

    const float epsF = static_cast<float>(eps);
    const float rmsScale = sqrtf(static_cast<float>(C));

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "scatter_gather3d_rmsnorm_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                const scalar_t *yData = y.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *gammaData = gamma.data_ptr<scalar_t>();
                const scalar_t *biasData = nullptr;
                if (bias.has_value()) {
                    TORCH_CHECK(bias.value().is_cuda(), "scatter_gather3d_rmsnorm_cuda: bias must be CUDA");
                    TORCH_CHECK(bias.value().is_contiguous(), "scatter_gather3d_rmsnorm_cuda: bias must be contiguous");
                    TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "scatter_gather3d_rmsnorm_cuda: bias dtype must match x");
                    TORCH_CHECK(bias.value().numel() == C, "scatter_gather3d_rmsnorm_cuda: bias.numel() must equal C");
                    biasData = bias.value().data_ptr<scalar_t>();
                }

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleT = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "scatter_gather3d_rmsnorm_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "scatter_gather3d_rmsnorm_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 5, "scatter_gather3d_rmsnorm_cuda: scale must be 5D");
                    TORCH_CHECK(broadcastable(y, scale.value()), "scatter_gather3d_rmsnorm_cuda: scale not broadcastable to y");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleT = static_cast<int>(scale.value().size(2));
                    scaleH = static_cast<int>(scale.value().size(3));
                    scaleW = static_cast<int>(scale.value().size(4));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftT = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "scatter_gather3d_rmsnorm_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "scatter_gather3d_rmsnorm_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 5, "scatter_gather3d_rmsnorm_cuda: shift must be 5D");
                    TORCH_CHECK(broadcastable(y, shift.value()), "scatter_gather3d_rmsnorm_cuda: shift not broadcastable to y");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftT = static_cast<int>(shift.value().size(2));
                    shiftH = static_cast<int>(shift.value().size(3));
                    shiftW = static_cast<int>(shift.value().size(4));
                }

                const int64_t numVecs = static_cast<int64_t>(B) * numActive * T * Ro * So;
                const dim3 blocks(static_cast<unsigned int>(numVecs), 1, 1);

                const int normThreads = (C <= 32) ? 32 : (C <= 64) ? 64 : (C <= 128) ? 128 : rms_norm_threads;
                const size_t shmemBytes = static_cast<size_t>(C) * sizeof(float);

                scatter_gather3d_rmsnorm_cuda_kernel<scalar_t><<<blocks, normThreads, shmemBytes>>>(
                        numVecs, numActive,
                        B, C, T, H, W,
                        Rx, Sx,
                        Ro, So,
                        xData, yData, outputData,
                        activeIndicesData, scatterMapData,
                        gammaData,
                        biasData,
                        epsF,
                        rmsScale,
                        scaleData,
                        scaleB, scaleC, scaleT, scaleH, scaleW,
                        shiftData,
                        shiftB, shiftC, shiftT, shiftH, shiftW,
                        activationType,
                        activationFirst);
            });

    return output;
}
