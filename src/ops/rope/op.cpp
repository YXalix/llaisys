#include "op.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // Check shapes: out and in should have shape [seq_len, n_heads, head_dim]
    // pos_ids should have shape [seq_len]
    ASSERT(out->ndim() == 3, "RoPE: output tensor must be 3D [seq_len, n_heads, head_dim]");
    ASSERT(in->ndim() == 3, "RoPE: input tensor must be 3D [seq_len, n_heads, head_dim]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids tensor must be 1D [seq_len]");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");
    
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must match seq_len");
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    ASSERT(head_dim % 2 == 0, "RoPE: head dimension must be even");
    ASSERT(out->isContiguous() && in->isContiguous(), "RoPE: all tensors must be contiguous");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), 
                        seq_len, n_heads, head_dim, theta);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), 
                        seq_len, n_heads, head_dim, theta);
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
