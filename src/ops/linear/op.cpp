#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // 检查维度
    CHECK_ARGUMENT(in->ndim() == 2, "input must be 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "output must be 2D tensor");
    
    // 检查形状匹配
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    size_t weight_in_features = weight->shape()[1];
    
    CHECK_ARGUMENT(in_features == weight_in_features, "input features must match weight input features");
    CHECK_ARGUMENT(out->shape()[0] == batch_size, "output batch size must match input batch size");
    CHECK_ARGUMENT(out->shape()[1] == out_features, "output features must match weight output features");
    
    // 检查偏置（如果提供）
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1, "bias must be 1D tensor");
        CHECK_ARGUMENT(bias->shape()[0] == out_features, "bias size must match output features");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }
    
    // 检查连续性
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: all tensors must be contiguous.");

    // 计算步长（已经是以元素为单位）
    size_t in_stride = in->strides()[0];
    size_t weight_stride = weight->strides()[0];
    size_t out_stride = out->strides()[0];

    // 总是支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features,
                          in_stride, weight_stride, out_stride);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features,
                          in_stride, weight_stride, out_stride);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
