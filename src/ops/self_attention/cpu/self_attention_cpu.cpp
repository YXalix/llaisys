#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                    float scale, size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    
    // Handle grouped query attention: repeat k,v heads to match q heads
    size_t head_ratio = nhead / nkvhead;
    
    // For each head
    for (size_t h = 0; h < nhead; h++) {
        size_t kv_head = h / head_ratio;  // which kv head to use for this q head
        
        // For each query position
        for (size_t i = 0; i < seqlen; i++) {
            // Compute attention scores: Q @ K^T * scale
            float *scores = new float[total_len];
            
            for (size_t j = 0; j < total_len; j++) {
                float score = 0.0f;
                
                // Dot product between q[i,h,:] and k[j,kv_head,:]
                for (size_t k_dim = 0; k_dim < d; k_dim++) {
                    T q_val = q[i * nhead * d + h * d + k_dim];
                    T k_val = k[j * nkvhead * d + kv_head * d + k_dim];
                    
                    score += llaisys::utils::cast<float>(q_val) * llaisys::utils::cast<float>(k_val);
                }
                
                score *= scale;
                
                // Apply causal mask: set to -inf if j > i
                if (j > i) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                scores[j] = score;
            }
            
            // Apply softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j <= i; j++) {  // only consider non-masked positions
                max_score = std::max(max_score, scores[j]);
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                if (j <= i) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                } else {
                    scores[j] = 0.0f;  // masked positions
                }
            }
            
            // Normalize
            for (size_t j = 0; j < total_len; j++) {
                scores[j] /= sum_exp;
            }
            
            // Compute output: scores @ V
            for (size_t v_dim = 0; v_dim < dv; v_dim++) {
                float output = 0.0f;
                
                for (size_t j = 0; j < total_len; j++) {
                    T v_val = v[j * nkvhead * dv + kv_head * dv + v_dim];
                    output += scores[j] * llaisys::utils::cast<float>(v_val);
                }
                
                attn_val[i * nhead * dv + h * dv + v_dim] = llaisys::utils::cast<T>(output);
            }
            
            delete[] scores;
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                   llaisysDataType_t dtype, float scale,
                   size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_(reinterpret_cast<float *>(attn_val),
                       reinterpret_cast<const float *>(q),
                       reinterpret_cast<const float *>(k),
                       reinterpret_cast<const float *>(v),
                       scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                       reinterpret_cast<const llaisys::fp16_t *>(q),
                       reinterpret_cast<const llaisys::fp16_t *>(k),
                       reinterpret_cast<const llaisys::fp16_t *>(v),
                       scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                       reinterpret_cast<const llaisys::bf16_t *>(q),
                       reinterpret_cast<const llaisys::bf16_t *>(k),
                       reinterpret_cast<const llaisys::bf16_t *>(v),
                       scale, seqlen, total_len, nhead, nkvhead, d, dv);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}
