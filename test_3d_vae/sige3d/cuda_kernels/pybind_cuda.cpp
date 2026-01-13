#include <torch/extension.h>

torch::Tensor gather2d_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName,
        bool activationFirst);

torch::Tensor scatter2d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &residual);

torch::Tensor scatter_with_block_residual2d_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1);

torch::Tensor scatter_gather2d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName,
        bool activationFirst);

torch::Tensor get_scatter_map_cuda(
        int H, int W,
        int bSizeH, int bSizeW,
        int kSizeH, int kSizeW,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices);

torch::Tensor gather3d_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName,
        bool activationFirst);

torch::Tensor gather3d_rmsnorm_cuda(
        const torch::Tensor &x,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &gamma,
        const torch::optional<torch::Tensor> &bias,
        double eps,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName);

torch::Tensor scatter3d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices,
        const torch::optional<torch::Tensor> &residual);

torch::Tensor scatter_with_block_residual3d_cuda(
        const torch::Tensor &x0, const torch::Tensor &y0,
        const torch::Tensor &x1, const torch::Tensor &y1,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices0,
        const torch::Tensor &activeIndices1);

torch::Tensor scatter_gather3d_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int bSizeH, int bSizeW,
        const torch::Tensor &activeIndices,
        const torch::Tensor &scatterMap,
        const torch::optional<torch::Tensor> &scale,
        const torch::optional<torch::Tensor> &shift,
        const std::string &activationName,
        bool activationFirst);

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
        const std::string &activationName,
        bool activationFirst);

torch::Tensor rms_norm_cuda(
        const torch::Tensor &x,
        const torch::Tensor &gamma,
        const torch::optional<torch::Tensor> &bias,
        double eps,
        bool channelFirst);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SIGE3D CUDA kernels";

    m.def("gather2d", &gather2d_cuda, "Gather2d (CUDA)");
    m.def("scatter2d", &scatter2d_cuda, "Scatter2d (CUDA)");
    m.def("scatter_with_block_residual2d", &scatter_with_block_residual2d_cuda, "Scatter with block residual2d (CUDA)");
    m.def("scatter_gather2d", &scatter_gather2d_cuda, "Scatter-Gather2d (CUDA)");

    // shared H/W scatter map (used by both 2D/3D pipelines)
    m.def("get_scatter_map", &get_scatter_map_cuda, "Get scatter map (CUDA)");

    m.def("gather3d", &gather3d_cuda, "Gather3d (CUDA)");
    m.def("gather3d_rmsnorm", &gather3d_rmsnorm_cuda, "Gather3d with RMSNorm (CUDA)");
    m.def("scatter3d", &scatter3d_cuda, "Scatter3d (CUDA)");
    m.def("scatter_with_block_residual3d", &scatter_with_block_residual3d_cuda, "Scatter with block residual3d (CUDA)");
    m.def("scatter_gather3d", &scatter_gather3d_cuda, "Scatter-Gather3d (CUDA)");
    m.def("scatter_gather3d_rmsnorm", &scatter_gather3d_rmsnorm_cuda, "Scatter-Gather3d with RMSNorm (CUDA)");

    m.def("rms_norm", &rms_norm_cuda, "RMSNorm (CUDA)");
}
