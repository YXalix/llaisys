#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
            size_t batch_size, size_t in_features, size_t out_features,
            size_t in_stride, size_t weight_stride, size_t out_stride) {
    
    // Y = X * W^T + b
    // out[i,j] = sum(in[i,k] * weight[j,k]) + bias[j]
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum_f = 0;

            // 计算矩阵乘法 X * W^T
            for (size_t k = 0; k < in_features; k++) {
                T in_val = in[i * in_stride + k];
                T weight_val = weight[j * weight_stride + k];
                sum_f += llaisys::utils::cast<float>(in_val) * llaisys::utils::cast<float>(weight_val);
            }

            // 添加bias
            if (bias != nullptr) {
                float bias_f = llaisys::utils::cast<float>(bias[j]);
                sum_f += bias_f;
            }

            out[i * out_stride + j] = llaisys::utils::cast<T>(sum_f);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
           llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features,
           size_t in_stride, size_t weight_stride, size_t out_stride) {
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_(reinterpret_cast<float *>(out),
               reinterpret_cast<const float *>(in),
               reinterpret_cast<const float *>(weight),
               bias ? reinterpret_cast<const float *>(bias) : nullptr,
               batch_size, in_features, out_features,
               in_stride, weight_stride, out_stride);
        break;
    case LLAISYS_DTYPE_F16:
        linear_(reinterpret_cast<llaisys::fp16_t *>(out),
               reinterpret_cast<const llaisys::fp16_t *>(in),
               reinterpret_cast<const llaisys::fp16_t *>(weight),
               bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
               batch_size, in_features, out_features,
               in_stride, weight_stride, out_stride);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_(reinterpret_cast<llaisys::bf16_t *>(out),
               reinterpret_cast<const llaisys::bf16_t *>(in),
               reinterpret_cast<const llaisys::bf16_t *>(weight),
               bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
               batch_size, in_features, out_features,
               in_stride, weight_stride, out_stride);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}
