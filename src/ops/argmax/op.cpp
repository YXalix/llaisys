#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    
    // Check that max_idx and max_val are single-element tensors
    CHECK_ARGUMENT(max_idx->numel() == 1, "max_idx must be a single-element tensor");
    CHECK_ARGUMENT(max_val->numel() == 1, "max_val must be a single-element tensor");
    
    // Check that all tensors are contiguous
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), 
           "Argmax: all tensors must be contiguous.");

    // Always support CPU calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                          vals->dtype(), max_idx->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                          vals->dtype(), max_idx->dtype(), vals->numel());
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
