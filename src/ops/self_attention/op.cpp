#include "op.hpp"

#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "SelfAttention: all tensors must be contiguous.");

    // Extract dimensions
    ASSERT(q->ndim() == 3, "Query tensor must be 3D: [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "Key tensor must be 3D: [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "Value tensor must be 3D: [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "Output tensor must be 3D: [seqlen, nhead, dv]");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    ASSERT(k->shape()[2] == d, "Key dimension must match query dimension");
    ASSERT(v->shape()[0] == total_len && v->shape()[1] == nkvhead, "Value shape mismatch");
    ASSERT(attn_val->shape()[0] == seqlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == dv,
           "Output shape mismatch");
    ASSERT(nhead % nkvhead == 0, "Number of heads must be divisible by number of key/value heads");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), scale, seqlen, total_len, nhead, nkvhead, d, dv);
    }

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), scale, seqlen, total_len, nhead, nkvhead, d, dv);
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
