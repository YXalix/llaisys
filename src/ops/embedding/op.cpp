#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    std::cout << "Embedding operation 1" << std::endl;
    CHECK_SAME_DEVICE(out, index, weight);
    std::cout << "Embedding operation 2" << std::endl;
    // Check index dtype
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64 type");
    std::cout << "Embedding operation 3" << std::endl;
    
    // Check output and weight dtype match
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    std::cout << "Embedding operation 4" << std::endl;
    
    // Check that all tensors are contiguous
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous.");

    std::cout << "Embedding operation" << std::endl;

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
