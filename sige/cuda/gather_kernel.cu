#include "common_cuda.cu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

__global__ void gather_cuda_kernel(
        int total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        const float *__restrict__ x,
        float *__restrict__ output,
        const int *activeIndices,
        const float *__restrict__ scale,
        int scaleB, int scaleC, int scaleH, int scaleW,
        const float *__restrict__ shift,
        int shiftB, int shiftC, int shiftH, int shiftW,
        ActivationType activationType,
        bool activationFirst) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) // 如果我没有对应的元素，就下班。
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;

    // 把第 ib 个激活 block 的左上角坐标 (biH, biW)，加上 block 内偏移 (intraBh, intraBw)，
    // 映射回原始特征图 (H, W) 上的真实像素位置，并做越界保护。

    // activeIndices 里存的就是每个 active block 在原图上的左上角（top-left）坐标。
    int biH = activeIndices[ib << 1];
    int hh = biH + intraBh; // 就是在把 block 内坐标映射回原图的真实坐标。
    if (hh < 0 || hh >= H) {
        output[index] = 0;
        return;
    }
    int biW = activeIndices[ib << 1 | 1];
    int ww = biW + intraBw;
    if (ww < 0 || ww >= W) {
        output[index] = 0;
        return;
    }

    auto p = bb * C * H * W + cc * H * W + hh * W + ww;
    auto z = x[p];
    if (!activationFirst) {
        z = binary_op_array_cuda<MUL>(
                scale, z,
                scaleB, scaleC, scaleH, scaleW,
                bb, cc, hh, ww);
        z = binary_op_array_cuda<ADD>(
                shift, z,
                shiftB, shiftC, shiftH, shiftW,
                bb, cc, hh, ww);
    }
    z = activation_cuda(activationType, z);
    if (activationFirst) {
        z = binary_op_array_cuda<MUL>(
                scale, z,
                scaleB, scaleC, scaleH, scaleW,
                bb, cc, hh, ww);
        z = binary_op_array_cuda<ADD>(
                shift, z,
                shiftB, shiftC, shiftH, shiftW,
                bb, cc, hh, ww);
    }
    output[index] = z;
}


// gather_cuda 是一个 CUDA kernel 的“胶水层（glue code / launcher）”，
// 负责把 PyTorch 世界的 Tensor，翻译成 CUDA 世界的指针 + 整数参数。
torch::Tensor gather_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"),
        bool activationFirst = false) {
    const int R = bSizeH, S = bSizeW;
    const int numActive = activeIndices.size(0);

    // 我接下来要创建一个新 Tensor，它的数据类型和 x 一样，放在和 x 一样的设备上，而且它只是个普通结果，不参与反向传播。
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto xData = x.data_ptr<float>();

    const auto activeIndicesData = activeIndices.data_ptr<int>();

    const int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto output = torch::empty({B * numActive, C, R, S}, options);
    auto outputData = output.data_ptr<float>();

    const float *scaleData = nullptr;
    int scaleB = 0, scaleC = 0, scaleH = 0, scaleW = 0;
    if (scale.has_value()) {
        assert(broadcastable(x, scale.value()));
        scaleData = scale.value().data_ptr<float>();
        scaleB = scale.value().size(0);
        scaleC = scale.value().size(1);
        scaleH = scale.value().size(2);
        scaleW = scale.value().size(3);
    }

    const float *shiftData = nullptr;
    int shiftB = 0, shiftC = 0, shiftH = 0, shiftW = 0;
    if (shift.has_value()) {
        assert(broadcastable(x, shift.value()));
        shiftData = shift.value().data_ptr<float>();
        shiftB = shift.value().size(0);
        shiftC = shift.value().size(1);
        shiftH = shift.value().size(2);
        shiftW = shift.value().size(3);
    }

    const auto activationType = getActivationType(activationName);

    // 这就是“一共要算多少个 output 元素”
    const int total = output.numel();   // 783 * 16 * 6 * 6 = 451008

    // 上采样
    const dim3 blocks((total + threads - 1) / threads, 1);
    gather_cuda_kernel<<<blocks, threads>>>(
            total, numActive,
            B, C, H, W, R, S,
            xData, outputData, activeIndicesData,
            scaleData,
            scaleB, scaleC, scaleH, scaleW,
            shiftData,
            shiftB, shiftC, shiftH, shiftW,
            activationType, activationFirst);

    return output;
}


/*
在 C 语言直觉下: 
✔️ 4 重循环
✔️ 每次算 1 个 output 元素
✔️ index 是线性展开后的编号

for (int bb = 0; bb < B * numActive; bb++) {
    for (int cc = 0; cc < C; cc++) {
        for (int hh = 0; hh < R; hh++) {
            for (int ww = 0; ww < S; ww++) {

                int index = ((bb * C + cc) * R + hh) * S + ww;

                output[index] = ...; // gather + scale + shift + activation
            }
        }
    }
}

CUDA 不是不循环，而是：
把「for 循环的每一次迭代，交给一个线程来做」

CPU C: 串行
CUDA: 并行
*/