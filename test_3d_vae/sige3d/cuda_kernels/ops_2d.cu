#include "common_cuda.cu"

#include <limits>
#include <torch/extension.h>

template <typename scalar_t, typename index_t>
__global__ void gather2d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
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

    int64_t p = static_cast<int64_t>(bb) * C * H * W + static_cast<int64_t>(cc) * H * W + static_cast<int64_t>(hh) * W + ww;
    float z = to_float(x[p]);
    if (!activationFirst) {
        z = binary_op_array_cuda_4d<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_cuda_4d<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }
    z = activation_cuda(activationType, z);
    if (activationFirst) {
        z = binary_op_array_cuda_4d<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_cuda_4d<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }
    output[index] = from_float<scalar_t>(z);
}

torch::Tensor gather2d_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false
) {
    TORCH_CHECK(x.is_cuda(), "gather2d_cuda: x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "gather2d_cuda: x must be 4D [B,C,H,W]");
    TORCH_CHECK(activeIndices.is_cuda(), "gather2d_cuda: activeIndices must be CUDA");
    TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "gather2d_cuda: activeIndices must be int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "gather2d_cuda: activeIndices must be [N,2]");

    const int R = bSizeH, S = bSizeW;
    const int numActive = static_cast<int>(activeIndices.size(0));
    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, R, S}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const auto activationType = getActivationType(activationName);

    const int *activeIndicesData = activeIndices.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "gather2d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "gather2d_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "gather2d_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 4, "gather2d_cuda: scale must be 4D");
                    TORCH_CHECK(broadcastable(x, scale.value()), "gather2d_cuda: scale not broadcastable to x");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleH = static_cast<int>(scale.value().size(2));
                    scaleW = static_cast<int>(scale.value().size(3));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "gather2d_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "gather2d_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 4, "gather2d_cuda: shift must be 4D");
                    TORCH_CHECK(broadcastable(x, shift.value()), "gather2d_cuda: shift not broadcastable to x");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftH = static_cast<int>(shift.value().size(2));
                    shiftW = static_cast<int>(shift.value().size(3));
                }

                const int64_t total = output.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    gather2d_cuda_kernel<scalar_t, int><<<blocks, threads>>>(
                            total32, numActive,
                            B, C, H, W,
                            R, S,
                            xData, outputData, activeIndicesData,
                            scaleData,
                            scaleB, scaleC, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftH, shiftW,
                            activationType, activationFirst);
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    gather2d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, H, W,
                            R, S,
                            xData, outputData, activeIndicesData,
                            scaleData,
                            scaleB, scaleC, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftH, shiftW,
                            activationType, activationFirst);
                }
            });

    return output;
}

template <typename scalar_t, typename index_t>
__global__ void scatter2d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const scalar_t *__restrict__ x,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const scalar_t *__restrict__ residual,
        int residualB, int residualC, int residualH, int residualW
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;

    int64_t p = static_cast<int64_t>(bb) * C * H * W + static_cast<int64_t>(cc) * H * W + static_cast<int64_t>(hh) * W + ww;
    float z = to_float(x[index]);
    z = binary_op_array_cuda_4d<ADD>(residual, z, residualB, residualC, residualH, residualW, bb, cc, hh, ww);
    output[p] = from_float<scalar_t>(z);
}

template <typename scalar_t, typename index_t>
__global__ void calibrate_residual2d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const scalar_t *__restrict__ x,
        const scalar_t *__restrict__ y,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int cc = static_cast<int>(t % C);
    t /= C;
    int ib = static_cast<int>(t % numActive);
    int bb = static_cast<int>(t / numActive);

    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww >= W)
        return;

    int64_t p = static_cast<int64_t>(bb) * C * H * W + static_cast<int64_t>(cc) * H * W + static_cast<int64_t>(hh) * W + ww;
    float cur = to_float(output[p]);
    cur += to_float(x[index]) - to_float(y[p]);
    output[p] = from_float<scalar_t>(cur);
}

torch::Tensor scatter2d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &residual
) {
    TORCH_CHECK(x.is_cuda(), "scatter2d_cuda: x must be CUDA");
    TORCH_CHECK(y.is_cuda(), "scatter2d_cuda: y must be CUDA");
    TORCH_CHECK(x.dim() == 4, "scatter2d_cuda: x must be 4D [B*numActive,C,R,S]");
    TORCH_CHECK(y.dim() == 4, "scatter2d_cuda: y must be 4D [B,C,H,W]");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "scatter2d_cuda: x/y dtype must match");
    TORCH_CHECK(activeIndices.is_cuda(), "scatter2d_cuda: activeIndices must be CUDA");
    TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "scatter2d_cuda: activeIndices must be int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "scatter2d_cuda: activeIndices must be [N,2]");

    const int numActive = static_cast<int>(activeIndices.size(0));
    auto output = y.clone();
    if (numActive == 0 || x.numel() == 0) {
        return output;
    }

    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int H = static_cast<int>(y.size(2));
    const int W = static_cast<int>(y.size(3));

    TORCH_CHECK(x.size(0) == B * numActive, "scatter2d_cuda: x.size(0) must equal B*numActive");
    TORCH_CHECK(x.size(1) == C, "scatter2d_cuda: x.size(1) must equal C");

    const int R = static_cast<int>(x.size(2));
    const int S = static_cast<int>(x.size(3));

    const int *activeIndicesData = activeIndices.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x.scalar_type(),
            "scatter2d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *residualData = nullptr;
                int residualB = 0, residualC = 0, residualH = 0, residualW = 0;
                if (residual.has_value()) {
                    TORCH_CHECK(residual.value().is_cuda(), "scatter2d_cuda: residual must be CUDA");
                    TORCH_CHECK(residual.value().scalar_type() == x.scalar_type(), "scatter2d_cuda: residual dtype must match x");
                    TORCH_CHECK(residual.value().dim() == 4, "scatter2d_cuda: residual must be 4D");
                    TORCH_CHECK(broadcastable(y, residual.value()), "scatter2d_cuda: residual not broadcastable to y");
                    residualData = residual.value().data_ptr<scalar_t>();
                    residualB = static_cast<int>(residual.value().size(0));
                    residualC = static_cast<int>(residual.value().size(1));
                    residualH = static_cast<int>(residual.value().size(2));
                    residualW = static_cast<int>(residual.value().size(3));
                }

                const int64_t total = x.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    scatter2d_cuda_kernel<scalar_t, int><<<blocks, threads>>>(
                            total32, numActive,
                            B, C, H, W,
                            R, S,
                            offsetH, offsetW,
                            strideH, strideW,
                            xData, outputData,
                            activeIndicesData,
                            residualData,
                            residualB, residualC, residualH, residualW);
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    scatter2d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, H, W,
                            R, S,
                            offsetH, offsetW,
                            strideH, strideW,
                            xData, outputData,
                            activeIndicesData,
                            residualData,
                            residualB, residualC, residualH, residualW);
                }
            });

    return output;
}

torch::Tensor scatter_with_block_residual2d_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1
) {
    auto output = scatter2d_cuda(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);
    if (x1.numel() == 0 || activeIndices1.numel() == 0) {
        return output;
    }

    TORCH_CHECK(x1.is_cuda() && y1.is_cuda(), "scatter_with_block_residual2d_cuda: x1/y1 must be CUDA");
    TORCH_CHECK(x1.scalar_type() == y1.scalar_type(), "scatter_with_block_residual2d_cuda: x1/y1 dtype must match");
    TORCH_CHECK(x1.scalar_type() == output.scalar_type(), "scatter_with_block_residual2d_cuda: dtype mismatch");
    TORCH_CHECK(activeIndices1.is_cuda() && activeIndices1.scalar_type() == torch::kInt32, "scatter_with_block_residual2d_cuda: activeIndices1 must be CUDA int32");

    const int B = static_cast<int>(y1.size(0));
    const int C = static_cast<int>(y1.size(1));
    const int H = static_cast<int>(y1.size(2));
    const int W = static_cast<int>(y1.size(3));

    const int numActive = static_cast<int>(activeIndices1.size(0));
    TORCH_CHECK(x1.size(0) == B * numActive, "scatter_with_block_residual2d_cuda: x1.size(0) must equal B*numActive1");
    TORCH_CHECK(x1.size(1) == C, "scatter_with_block_residual2d_cuda: x1.size(1) must equal C");

    const int R = static_cast<int>(x1.size(2));
    const int S = static_cast<int>(x1.size(3));

    const int *activeIndicesData = activeIndices1.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            x1.scalar_type(),
            "scatter_with_block_residual2d_cuda",
            [&] {
                const scalar_t *x1Data = x1.data_ptr<scalar_t>();
                const scalar_t *y1Data = y1.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const int64_t total = x1.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    calibrate_residual2d_cuda_kernel<scalar_t, int><<<blocks, threads>>>(
                            total32, numActive,
                            B, C, H, W,
                            R, S,
                            x1Data, y1Data,
                            outputData,
                            activeIndicesData);
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    calibrate_residual2d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, H, W,
                            R, S,
                            x1Data, y1Data,
                            outputData,
                            activeIndicesData);
                }
            });

    return output;
}

template <typename scalar_t, typename index_t>
__global__ void scatter_gather2d_cuda_kernel(
        index_t total, int numActive,
        int B, int C, int H, int W,
        int Rx, int Sx,
        int Ro, int So,
        const scalar_t *__restrict__ x,
        const scalar_t *__restrict__ y,
        scalar_t *__restrict__ output,
        const int *__restrict__ activeIndices,
        const int *__restrict__ scatterMap,
        const scalar_t *__restrict__ scale,
        int scaleB, int scaleC, int scaleH, int scaleW,
        const scalar_t *__restrict__ shift,
        int shiftB, int shiftC, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % So);
    t /= So;
    int intraBh = static_cast<int>(t % Ro);
    t /= Ro;
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
    int64_t p = static_cast<int64_t>(bb) * C * H * W + static_cast<int64_t>(cc) * H * W + static_cast<int64_t>(hh) * W + ww;

    float z;
    if (bx >= 0) {
        int hx = scatterMap[scatterMapIndex + 1];
        int wx = scatterMap[scatterMapIndex + 2];
        int64_t px = (static_cast<int64_t>(bb) * numActive + bx) * C * Rx * Sx + static_cast<int64_t>(cc) * Rx * Sx + hx * Sx + wx;
        z = to_float(x[px]);
    } else {
        z = to_float(y[p]);
    }

    if (!activationFirst) {
        z = binary_op_array_cuda_4d<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_cuda_4d<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }
    z = activation_cuda(activationType, z);
    if (activationFirst) {
        z = binary_op_array_cuda_4d<MUL>(scale, z, scaleB, scaleC, scaleH, scaleW, bb, cc, hh, ww);
        z = binary_op_array_cuda_4d<ADD>(shift, z, shiftB, shiftC, shiftH, shiftW, bb, cc, hh, ww);
    }

    output[index] = from_float<scalar_t>(z);
}

template <typename index_t>
__global__ void get_scatter_map_cuda_kernel(
        index_t total,
        int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        int *__restrict__ output,
        const int *__restrict__ activeIndices
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) + static_cast<index_t>(threadIdx.x);
    if (index >= total)
        return;

    index_t t = index;
    int intraBw = static_cast<int>(t % S);
    t /= S;
    int intraBh = static_cast<int>(t % R);
    t /= R;
    int ib = static_cast<int>(t);

    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;

    int p = 3 * (hh * W + ww);
    output[p] = ib;
    output[p + 1] = intraBh;
    output[p + 2] = intraBw;
}

torch::Tensor scatter_gather2d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false
) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "scatter_gather2d_cuda: x/y must be CUDA");
    TORCH_CHECK(x.dim() == 4 && y.dim() == 4, "scatter_gather2d_cuda: x/y must be 4D");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "scatter_gather2d_cuda: x/y dtype must match");
    TORCH_CHECK(activeIndices.is_cuda() && activeIndices.scalar_type() == torch::kInt32, "scatter_gather2d_cuda: activeIndices must be CUDA int32");
    TORCH_CHECK(scatterMap.is_cuda() && scatterMap.scalar_type() == torch::kInt32, "scatter_gather2d_cuda: scatterMap must be CUDA int32");

    const int Ro = bSizeH, So = bSizeW;
    const int Rx = static_cast<int>(x.size(2));
    const int Sx = static_cast<int>(x.size(3));
    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int H = static_cast<int>(y.size(2));
    const int W = static_cast<int>(y.size(3));

    const int numActive = static_cast<int>(activeIndices.size(0));
    TORCH_CHECK(x.size(1) == C, "scatter_gather2d_cuda: x.size(1) must equal C");
    TORCH_CHECK(x.size(0) == B * numActive, "scatter_gather2d_cuda: x.size(0) must equal B*numActive");
    TORCH_CHECK(scatterMap.size(0) == H && scatterMap.size(1) == W && scatterMap.size(2) == 3, "scatter_gather2d_cuda: scatterMap must be [H,W,3]");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, Ro, So}, options);
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
            "scatter_gather2d_cuda",
            [&] {
                const scalar_t *xData = x.data_ptr<scalar_t>();
                const scalar_t *yData = y.data_ptr<scalar_t>();
                scalar_t *outputData = output.data_ptr<scalar_t>();

                const scalar_t *scaleData = nullptr;
                int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
                if (scale.has_value()) {
                    TORCH_CHECK(scale.value().is_cuda(), "scatter_gather2d_cuda: scale must be CUDA");
                    TORCH_CHECK(scale.value().scalar_type() == x.scalar_type(), "scatter_gather2d_cuda: scale dtype must match x");
                    TORCH_CHECK(scale.value().dim() == 4, "scatter_gather2d_cuda: scale must be 4D");
                    TORCH_CHECK(broadcastable(y, scale.value()), "scatter_gather2d_cuda: scale not broadcastable to y");
                    scaleData = scale.value().data_ptr<scalar_t>();
                    scaleB = static_cast<int>(scale.value().size(0));
                    scaleC = static_cast<int>(scale.value().size(1));
                    scaleH = static_cast<int>(scale.value().size(2));
                    scaleW = static_cast<int>(scale.value().size(3));
                }

                const scalar_t *shiftData = nullptr;
                int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
                if (shift.has_value()) {
                    TORCH_CHECK(shift.value().is_cuda(), "scatter_gather2d_cuda: shift must be CUDA");
                    TORCH_CHECK(shift.value().scalar_type() == x.scalar_type(), "scatter_gather2d_cuda: shift dtype must match x");
                    TORCH_CHECK(shift.value().dim() == 4, "scatter_gather2d_cuda: shift must be 4D");
                    TORCH_CHECK(broadcastable(y, shift.value()), "scatter_gather2d_cuda: shift not broadcastable to y");
                    shiftData = shift.value().data_ptr<scalar_t>();
                    shiftB = static_cast<int>(shift.value().size(0));
                    shiftC = static_cast<int>(shift.value().size(1));
                    shiftH = static_cast<int>(shift.value().size(2));
                    shiftW = static_cast<int>(shift.value().size(3));
                }

                const int64_t total = output.numel();
                if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    const int total32 = static_cast<int>(total);
                    const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
                    scatter_gather2d_cuda_kernel<scalar_t, int><<<blocks, threads>>>(
                            total32, numActive,
                            B, C, H, W,
                            Rx, Sx,
                            Ro, So,
                            xData, yData, outputData,
                            activeIndicesData, scatterMapData,
                            scaleData,
                            scaleB, scaleC, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftH, shiftW,
                            activationType, activationFirst);
                } else {
                    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
                    scatter_gather2d_cuda_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                            total, numActive,
                            B, C, H, W,
                            Rx, Sx,
                            Ro, So,
                            xData, yData, outputData,
                            activeIndicesData, scatterMapData,
                            scaleData,
                            scaleB, scaleC, scaleH, scaleW,
                            shiftData,
                            shiftB, shiftC, shiftH, shiftW,
                            activationType, activationFirst);
                }
            });

    return output;
}

torch::Tensor get_scatter_map_cuda(
        int H, int W,
        int bSizeH, int bSizeW,
        int kSizeH, int kSizeW,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices
) {
    TORCH_CHECK(activeIndices.is_cuda(), "get_scatter_map_cuda: activeIndices must be CUDA");
    TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "get_scatter_map_cuda: activeIndices must be int32");
    TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "get_scatter_map_cuda: activeIndices must be [N,2]");

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(activeIndices.device()).requires_grad(false);
    auto scatterMap = torch::full({H, W, 3}, -1, options);

    const int R = (bSizeH - kSizeH) / strideH + 1;
    const int S = (bSizeW - kSizeW) / strideW + 1;

    const int numActive = static_cast<int>(activeIndices.size(0));
    if (numActive == 0) {
        return scatterMap;
    }

    const int64_t total = static_cast<int64_t>(numActive) * R * S;
    if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
        const int total32 = static_cast<int>(total);
        const dim3 blocks(static_cast<unsigned int>((total32 + threads - 1) / threads), 1, 1);
        get_scatter_map_cuda_kernel<int><<<blocks, threads>>>(
                total32,
                H, W,
                R, S,
                offsetH, offsetW,
                strideH, strideW,
                scatterMap.data_ptr<int>(),
                activeIndices.data_ptr<int>());
    } else {
        const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads), 1, 1);
        get_scatter_map_cuda_kernel<int64_t><<<blocks, threads>>>(
                total,
                H, W,
                R, S,
                offsetH, offsetW,
                strideH, strideW,
                scatterMap.data_ptr<int>(),
                activeIndices.data_ptr<int>());
    }

    return scatterMap;
}
