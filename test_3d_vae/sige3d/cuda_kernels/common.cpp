#pragma once

#include <torch/extension.h>

#include <stdexcept>
#include <string>

// Keep as header-like .cpp for simple inclusion in both .cu and .cpp compilation units.

enum OpType {
    ADD,
    MUL,
};

enum ActivationType {
    IDENTITY,
    SWISH,
    RELU,
    SIGMOID,
    TANH,
};

inline ActivationType getActivationType(const std::string &activationName) {
    // NOTE: Avoid global std::string definitions in this header-like file, since it is
    // included by multiple CUDA translation units (would cause duplicate symbols).
    if (activationName == "identity")
        return IDENTITY;
    if (activationName == "swish" || activationName == "silu")
        return SWISH;
    if (activationName == "relu")
        return RELU;
    if (activationName == "sigmoid")
        return SIGMOID;
    if (activationName == "tanh")
        return TANH;
    throw std::invalid_argument("Unknown activation: " + activationName);
}

inline bool broadcastable(const torch::Tensor &x, const torch::Tensor &y) {
    auto xSize = x.sizes();
    auto ySize = y.sizes();
    if (xSize.size() != ySize.size())
        return false;
    for (int i = 0; i < static_cast<int>(xSize.size()); ++i) {
        if (xSize[i] != ySize[i] && xSize[i] != 1 && ySize[i] != 1)
            return false;
    }
    return true;
}
