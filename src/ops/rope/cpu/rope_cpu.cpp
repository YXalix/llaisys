#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, 
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // Iterate over sequence length
    for (size_t i = 0; i < seq_len; i++) {
        int64_t pos = pos_ids[i];  // Position ID for this token
        
        // Iterate over heads
        for (size_t h = 0; h < n_heads; h++) {
            // Calculate base indices for this sequence position and head
            size_t base_idx = i * n_heads * head_dim + h * head_dim;
            
            // Apply RoPE to each pair of dimensions
            for (size_t j = 0; j < head_dim / 2; j++) {
                // Calculate angle: phi_{i,j} = p_i / theta^{2j/d}
                float a_val, b_val, a_new, b_new;
                float angle = pos / powf(theta, 2.0f * j / head_dim);
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                size_t a_idx = base_idx + j;
                size_t b_idx = base_idx + j + head_dim / 2;

                a_val = llaisys::utils::cast<float>(in[a_idx]);
                b_val = llaisys::utils::cast<float>(in[b_idx]);
                
                // Apply rotation:
                // a'_{i,j} = a_{i,j} * cos(phi_{i,j}) - b_{i,j} * sin(phi_{i,j})
                // b'_{i,j} = b_{i,j} * cos(phi_{i,j}) + a_{i,j} * sin(phi_{i,j})
                a_new = a_val * cos_val - b_val * sin_val;
                b_new = b_val * cos_val + a_val * sin_val;

                out[a_idx] = llaisys::utils::cast<T>(a_new);
                out[b_idx] = llaisys::utils::cast<T>(b_new);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), 
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), 
                     pos_ids_ptr, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
