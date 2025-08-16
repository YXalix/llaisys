#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Check index dtype
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64 type");
    // Check output and weight dtype match
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    // Check that all tensors are contiguous
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous.");

    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                             weight->dtype(), index->numel(), weight->shape()[1], 
                             weight->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                             weight->dtype(), index->numel(), weight->shape()[1], 
                             weight->elementSize());
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
