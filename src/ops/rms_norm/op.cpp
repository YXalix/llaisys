#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMSNorm: all tensors must be contiguous.");
    
    // Check dimensions
    ASSERT(out->ndim() == 2 && in->ndim() == 2, "RMSNorm: input and output must be 2D tensors.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D tensor.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(weight->shape()[0] == in->shape()[1], 
           "RMSNorm: weight length must match input's last dimension.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), rows, cols, eps);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), rows, cols, eps);
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
