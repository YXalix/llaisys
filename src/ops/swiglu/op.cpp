#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), 
           "SwiGLU: all tensors must be contiguous.");
    
    // Check dimensions
    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2, 
           "SwiGLU: all tensors must be 2D.");
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    size_t numel = out->numel();

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                          out->dtype(), numel);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), 
                          out->dtype(), numel);
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
