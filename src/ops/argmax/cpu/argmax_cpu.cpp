#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T, typename IdxT>
void argmax_(IdxT *max_idx, T *max_val, const T *vals, size_t size) {
    if (size == 0) return;
    
    T max_value = vals[0];
    IdxT max_index = 0;
    
    for (size_t i = 1; i < size; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            T current_val = llaisys::utils::cast<T>(llaisys::utils::cast<float>(vals[i]));
            T max_val_converted = llaisys::utils::cast<T>(llaisys::utils::cast<float>(max_value));
            if (llaisys::utils::cast<float>(current_val) > llaisys::utils::cast<float>(max_val_converted)) {
                max_value = current_val;
                max_index = static_cast<IdxT>(i);
            }
        } else {
            if (vals[i] > max_value) {
                max_value = vals[i];
                max_index = static_cast<IdxT>(i);
            }
        }
    }
    
    *max_idx = max_index;
    *max_val = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
           llaisysDataType_t vals_type, llaisysDataType_t idx_type, size_t size) {
    
    // Handle different value types
    switch (vals_type) {
    case LLAISYS_DTYPE_F32:
        if (idx_type == LLAISYS_DTYPE_I64) {
            argmax_(reinterpret_cast<int64_t *>(max_idx), 
                   reinterpret_cast<float *>(max_val),
                   reinterpret_cast<const float *>(vals), size);
        } else {
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
        }
        break;
    case LLAISYS_DTYPE_F16:
        if (idx_type == LLAISYS_DTYPE_I64) {
            argmax_(reinterpret_cast<int64_t *>(max_idx), 
                   reinterpret_cast<llaisys::fp16_t *>(max_val),
                   reinterpret_cast<const llaisys::fp16_t *>(vals), size);
        } else {
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
        }
        break;
    case LLAISYS_DTYPE_BF16:
        if (idx_type == LLAISYS_DTYPE_I64) {
            argmax_(reinterpret_cast<int64_t *>(max_idx), 
                   reinterpret_cast<llaisys::bf16_t *>(max_val),
                   reinterpret_cast<const llaisys::bf16_t *>(vals), size);
        } else {
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
        }
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals_type);
    }
}
}
