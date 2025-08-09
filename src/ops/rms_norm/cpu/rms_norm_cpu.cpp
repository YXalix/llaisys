#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t row = 0; row < rows; row++) {
        const T *in_row = in + row * cols;
        T *out_row = out + row * cols;
        
        // Calculate sum of squares for this row
        float sum_squares = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float val = llaisys::utils::cast<float>(in_row[col]);
            sum_squares += val * val;
        }
        
        // Calculate RMS normalization factor
        float rms = sqrtf(sum_squares / cols + eps);
        float inv_rms = 1.0f / rms;
        
        // Apply normalization and weight
        for (size_t col = 0; col < cols; col++) {
            float in_val, weight_val, result;
            in_val = llaisys::utils::cast<float>(in_row[col]);
            weight_val = llaisys::utils::cast<float>(weight[col]);
            result = weight_val * in_val * inv_rms;
            out_row[col] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in), 
                        reinterpret_cast<const float *>(weight), 
                        rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), 
                        rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight), 
                        rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
