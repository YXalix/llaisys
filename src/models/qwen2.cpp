#include "qwen2.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "../tensor/tensor.hpp"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/add/op.hpp"

namespace llaisys {
    struct LlaisysQwen2Model *Qwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        std::cout << "Creating Qwen2 model with meta: " << meta->nlayer << " layers, " 
                  << meta->hs << " hidden size, " << meta->nh << " heads, "
                  << meta->nkvh << " KV heads, " << meta->dh << " head dimension, "
                  << meta->di << " intermediate dimension, max sequence length: "
                  << meta->maxseq << ", vocabulary size: " << meta->voc << std::endl;
        if (!meta) {
            std::cerr << "Invalid meta pointer" << std::endl;
            return nullptr;
        }
        struct LlaisysQwen2Model *model = new (std::nothrow) LlaisysQwen2Model();
        if (!model) {
            return nullptr;
        }
        std::cout << "Model device: " << model->device << std::endl;
        model->device = device;
        model->ndevice = ndevice;
        model->device_ids = new int[ndevice];
        if (!model->device_ids) {
            delete model;
            return nullptr;
        }
        std::copy(device_ids, device_ids + ndevice, model->device_ids);
        // 复制元数据
        model->meta = meta;
        // 初始化权重结构
        model->weights = {}; // Initialize weights to default values
        model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];
        return model;
    }

    void Qwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        std::cout << "Destroying Qwen2 model" << std::endl;
        if (!model) return;
        delete[] model->device_ids;
        delete[] model->weights.attn_norm_w;
        delete[] model->weights.attn_q_w;
        delete[] model->weights.attn_q_b;
        delete[] model->weights.attn_k_w;
        delete[] model->weights.attn_k_b;
        delete[] model->weights.attn_v_w;
        delete[] model->weights.attn_v_b;
        delete[] model->weights.attn_o_w;
        delete[] model->weights.mlp_norm_w;
        delete[] model->weights.mlp_gate_w;
        delete[] model->weights.mlp_up_w;
        delete[] model->weights.mlp_down_w;
        delete model;
    }

    struct LlaisysQwen2Weights *Qwen2ModelWeights(struct LlaisysQwen2Model *model) {
        std::cout << "Getting Qwen2 model weights" << std::endl;
        if (!model) return nullptr;
        return &model->weights;
    }

    int64_t Qwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !token_ids || ntoken == 0) return -1;
        
        std::cout << "Qwen2 inference with " << ntoken << " tokens" << std::endl;
        
        const auto& meta = model->meta;
        const auto& weights = model->weights;
        
        try {
            // 1. 输入嵌入层 - Input Embedding
            // 创建输入token tensor
            auto input_ids = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, model->device);
            std::memcpy(input_ids->data(), token_ids, ntoken * sizeof(int64_t));
            
            // 创建embedding输出tensor [ntoken, hidden_size]
            auto x = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            
            // 应用输入嵌入
            auto in_embed_tensor = weights.in_embed->tensor;
            llaisys::ops::embedding(x, input_ids, in_embed_tensor);
            
            // 预分配中间张量以减少内存分配开销
            auto attn_input = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            auto attn_residual = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            auto mlp_input = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            auto layer_output = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            
            // 预分配注意力相关张量
            auto q = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            auto k = llaisys::Tensor::create({ntoken, meta->nkvh * meta->dh}, meta->dtype, model->device);
            auto v = llaisys::Tensor::create({ntoken, meta->nkvh * meta->dh}, meta->dtype, model->device);
            auto attn_proj = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            
            // 预分配位置编码张量（在所有层中复用）
            auto pos_ids = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, model->device);
            int64_t* pos_data = reinterpret_cast<int64_t*>(pos_ids->data());
            for (size_t i = 0; i < ntoken; ++i) {
                pos_data[i] = static_cast<int64_t>(i);
            }
            
            // 预分配RoPE相关张量
            auto q_rope_3d = llaisys::Tensor::create({ntoken, meta->nh, meta->dh}, meta->dtype, model->device);
            auto k_rope_3d = llaisys::Tensor::create({ntoken, meta->nkvh, meta->dh}, meta->dtype, model->device);
            auto attn_output_3d = llaisys::Tensor::create({ntoken, meta->nh, meta->dh}, meta->dtype, model->device);

            size_t mlp_hidden_dim = weights.mlp_gate_w[0]->tensor->shape()[0];
            auto gate = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
            auto up = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
            auto mlp_hidden = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
            auto mlp_output = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);

            // 2. Transformer层
            for (size_t layer = 0; layer < meta->nlayer; ++layer) {
                // 获取当前层的权重
                auto attn_norm_w = weights.attn_norm_w[layer]->tensor;
                auto attn_q_w = weights.attn_q_w[layer]->tensor;
                auto attn_q_b = weights.attn_q_b[layer]->tensor;
                auto attn_k_w = weights.attn_k_w[layer]->tensor;
                auto attn_k_b = weights.attn_k_b[layer]->tensor;
                auto attn_v_w = weights.attn_v_w[layer]->tensor;
                auto attn_v_b = weights.attn_v_b[layer]->tensor;
                auto attn_o_w = weights.attn_o_w[layer]->tensor;
                auto mlp_norm_w = weights.mlp_norm_w[layer]->tensor;
                auto mlp_gate_w = weights.mlp_gate_w[layer]->tensor;
                auto mlp_up_w = weights.mlp_up_w[layer]->tensor;
                auto mlp_down_w = weights.mlp_down_w[layer]->tensor;
                
                // 2.1 Self-Attention
                // 注意力前的LayerNorm
                llaisys::ops::rms_norm(attn_input, x, attn_norm_w, meta->epsilon);
                
                // Q, K, V投影 - 复用预分配的张量
                llaisys::ops::linear(q, attn_input, attn_q_w, attn_q_b);
                llaisys::ops::linear(k, attn_input, attn_k_w, attn_k_b);
                llaisys::ops::linear(v, attn_input, attn_v_w, attn_v_b);
                
                // 将Q和K重塑为3D张量以适配RoPE操作
                // Q: [ntoken, hidden_size] -> [ntoken, n_heads, head_dim]
                auto q_3d = q->view({ntoken, meta->nh, meta->dh});
                
                // K: [ntoken, nkvh * head_dim] -> [ntoken, n_kv_heads, head_dim]
                auto k_3d = k->view({ntoken, meta->nkvh, meta->dh});
                
                llaisys::ops::rope(q_rope_3d, q_3d, pos_ids, meta->theta);
                llaisys::ops::rope(k_rope_3d, k_3d, pos_ids, meta->theta);
                
                // 自注意力计算 - 直接使用3D张量
                auto v_3d = v->view({ntoken, meta->nkvh, meta->dh});
                float scale = 1.0f / std::sqrt(static_cast<float>(meta->dh));
                llaisys::ops::self_attention(attn_output_3d, q_rope_3d, k_rope_3d, v_3d, scale);
                
                // 将注意力输出重塑回2D
                auto attn_output = attn_output_3d->view({ntoken, meta->hs});
                
                // 注意力输出投影 - 复用预分配的张量
                llaisys::ops::linear(attn_proj, attn_output, attn_o_w, nullptr);
                
                // 残差连接: attn_residual = x + attn_proj
                llaisys::ops::add(attn_residual, x, attn_proj);
                
                // 2.2 MLP
                // MLP前的LayerNorm
                llaisys::ops::rms_norm(mlp_input, attn_residual, mlp_norm_w, meta->epsilon);
                
                // MLP层：SwiGLU激活
                // 从权重矩阵形状推断正确的输出维度
                // size_t mlp_hidden_dim = mlp_gate_w->shape()[0];
                // auto gate = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
                // auto up = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
                
                llaisys::ops::linear(gate, mlp_input, mlp_gate_w, nullptr);
                llaisys::ops::linear(up, mlp_input, mlp_up_w, nullptr);
                // auto mlp_hidden = llaisys::Tensor::create({ntoken, mlp_hidden_dim}, meta->dtype, model->device);
                llaisys::ops::swiglu(mlp_hidden, gate, up);
                
                // MLP输出投影
                // auto mlp_output = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
                llaisys::ops::linear(mlp_output, mlp_hidden, mlp_down_w, nullptr);
                
                // 残差连接: x = attn_residual + mlp_output
                llaisys::ops::add(layer_output, attn_residual, mlp_output);
                
                // 更新x为下一层的输入
                x = layer_output;
            }
            
            // 3. 输出层
            // 最终LayerNorm
            auto out_norm_w = weights.out_norm_w->tensor;
            auto normed_output = llaisys::Tensor::create({ntoken, meta->hs}, meta->dtype, model->device);
            llaisys::ops::rms_norm(normed_output, x, out_norm_w, meta->epsilon);
            
            // 输出嵌入层 (Language Model Head)
            auto out_embed_tensor = weights.out_embed->tensor;
            auto logits = llaisys::Tensor::create({ntoken, meta->voc}, meta->dtype, model->device);
            llaisys::ops::linear(logits, normed_output, out_embed_tensor, nullptr);
            
            // 4. 取最后一个token的logits并进行argmax
            // 获取最后一个token的logits: [vocab_size]
            // 修复slice操作 - 直接获取最后一行
            auto last_logits = logits->slice(0, ntoken - 1, ntoken); // shape: [1, vocab_size]
            auto last_logits_1d = last_logits->view({meta->voc}); // shape: [vocab_size]
            
            // 进行argmax获取下一个token
            auto max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, model->device);
            auto max_val = llaisys::Tensor::create({1}, meta->dtype, model->device);
            
            llaisys::ops::argmax(max_idx, max_val, last_logits_1d);
            
            // 获取结果
            int64_t* result_data = reinterpret_cast<int64_t*>(max_idx->data());
            int64_t next_token = result_data[0];
            
            return next_token;
            
        } catch (const std::exception& e) {
            std::cerr << "Error during Qwen2 inference: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cerr << "Unknown error during Qwen2 inference" << std::endl;
            return -1;
        }
    }
}

