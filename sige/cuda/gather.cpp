#include <torch/extension.h>

torch::Tensor gather_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW, // 每个小块的高度和宽度
        const torch::Tensor &activeIndices, // 哪些地方要取
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName = std::string("identity"), // 激活函数的名字
        bool activationFirst = false);
