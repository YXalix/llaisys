from typing import Sequence
from ctypes import POINTER, cast
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download
import safetensors
from ..libllaisys.tensor import llaisysTensor_t

from ..tensor import Tensor
from ..libllaisys.models import LlaisysQwen2Meta
from ..libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)

def set_tensor_array(tensor_array_ptr, index: int, tensor: llaisysTensor_t, array_size: int):
    """给指针数组的特定位置赋值
    
    Args:
        tensor_array_ptr: 张量指针数组
        index: 要赋值的位置
        tensor: 要赋的张量
        array_size: 数组大小
    """
    if not tensor_array_ptr or index < 0 or index >= array_size:
        raise ValueError(f"Invalid tensor array or index {index}")
    
    # 将指针转换为数组并赋值
    array = cast(tensor_array_ptr, POINTER(llaisysTensor_t * array_size)).contents
    array[index] = tensor


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
            # 1. 构造 meta 信息
            self.meta = LlaisysQwen2Meta()
            # 使用 bfloat16 数据类型
            self.meta.dtype = llaisysDataType_t(DataType.BF16)
            # 从配置文件中获取模型结构参数
            self.meta.nlayer = 28                # num_hidden_layers
            self.meta.hs = 1536                  # hidden_size
            self.meta.nh = 12                    # num_attention_heads
            self.meta.nkvh = 2                   # num_key_value_heads
            self.meta.dh = self.meta.hs // self.meta.nh  # 每个注意力头的维度
            self.meta.di = self.meta.hs // self.meta.nh  # 中间层维度
            self.meta.maxseq = 4096              # sliding_window
            self.meta.voc = 151936               # vocab_size
            self.meta.epsilon = 1e-6             # rms_norm_eps
            self.meta.theta = 10000.0            # rope_theta
            self.meta.end_token = 151643         # eos_token_id

            print(f"Qwen2 Meta: {self.meta}")

            self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
                self.meta,
                llaisysDeviceType_t(device),
                None,  # device_id, default to 0
                0,     # stream, default to 0
            )

            # 2. 构造权重结构体（通常由C端分配，这里仅演示）
            # self.weight_tensor = LIB_LLAISYS.LlaisysQwen2Weights()

            if not self.model:
                raise RuntimeError("Failed to create Qwen2 model")

            # 获取权重结构
            weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
            if not weights_ptr:
                raise RuntimeError("Failed to get model weights")
            self.weights = weights_ptr
            
            # 保存所有tensor对象的引用，防止被垃圾回收
            self._tensor_refs = {}

            # 3. 加载权重
            model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
            for file in sorted(Path(model_dir).glob("*.safetensors")):
                print(f"Loading model weights from {file}")
                with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                    for name_ in f.keys():
                        torch_tensor = f.get_tensor(name_)
                        llaisys_tensor = Tensor(
                            shape=torch_tensor.shape,
                            dtype=self.meta.dtype,
                        )
                        llaisys_tensor.load(torch_tensor.data_ptr())
                        # print(f"Loaded tensor {name_} with shape {torch_tensor.shape} and dtype {torch_tensor.dtype}")
                        # 保存tensor引用，防止被垃圾回收
                        self._tensor_refs[name_] = llaisys_tensor
                        if name_ == "lm_head.weight":
                            self.weights.contents.out_embed = llaisys_tensor.lib_tensor()
                        elif name_ == "model.embed_tokens.weight":
                            self.weights.contents.in_embed = llaisys_tensor.lib_tensor()
                        elif "input_layernorm.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_norm_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "mlp.down_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.mlp_down_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "mlp.gate_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.mlp_gate_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "mlp.up_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.mlp_up_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "post_attention_layernorm.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.mlp_norm_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.k_proj.bias" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_k_b,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.k_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_k_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.o_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_o_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.q_proj.bias" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_q_b,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.q_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_q_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.v_proj.bias" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_v_b,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "self_attn.v_proj.weight" in name_:
                            layer_idx = int(name_.split(".")[2])
                            set_tensor_array(
                                self.weights.contents.attn_v_w,
                                layer_idx,
                                llaisys_tensor.lib_tensor(),
                                self.meta.nlayer,
                            )
                        elif "norm.weight" in name_:
                            self.weights.contents.out_norm_w = llaisys_tensor.lib_tensor()
                        else:
                            print(f"Unknown tensor {name_}")

    def __del__(self):
        """析构函数，确保资源正确释放"""
        if hasattr(self, 'model') and self.model:
            # 如果有相应的销毁函数，在这里调用
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

        if hasattr(self, '_tensor_refs'):
            # 清理tensor引用，让垃圾回收器处理
            self._tensor_refs.clear()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """
        生成文本序列
        
        Args:
            inputs: 输入token序列
            max_new_tokens: 最大生成的新token数量，默认为128
            top_k: top-k采样的k值
            top_p: nucleus采样的p值
            temperature: 温度参数，控制随机性
            
        Returns:
            List[int]: 生成的完整token序列（包括输入）
        """
        import random
        import numpy as np
        from ctypes import c_int64, POINTER
        
        if max_new_tokens is None:
            max_new_tokens = 128
            
        # 检查输入长度是否超过最大序列长度
        if len(inputs) >= self.meta.maxseq:
            print(f"Warning: Input length {len(inputs)} exceeds max sequence length {self.meta.maxseq}")
            inputs = inputs[-self.meta.maxseq+1:]  # 保留最后的部分，留出空间给新token
            
        # 将输入转换为token列表
        generated_tokens = list(inputs)
        
        print(f"Starting generation with {len(inputs)} input tokens, max_new_tokens={max_new_tokens}")
        
        for step in range(max_new_tokens):
            # 检查是否超过最大序列长度
            if len(generated_tokens) >= self.meta.maxseq:
                print(f"Reached maximum sequence length {self.meta.maxseq}")
                break
                
            # 准备输入token数组
            current_length = len(generated_tokens)
            token_array = (c_int64 * current_length)(*generated_tokens)
            
            # 调用底层推理函数
            try:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    token_array,
                    current_length
                )
                
                # 检查返回值
                if next_token < 0:
                    print(f"Error during inference at step {step}")
                    break
                    
                # 简单的贪心采样（当前底层函数可能只返回最可能的token）
                # 在未来的实现中，这里应该接收完整的logits并进行采样
                if top_k == 1:
                    # 贪心采样
                    selected_token = next_token
                else:
                    # 目前底层函数只返回一个token，所以直接使用
                    # 在完整实现中，这里应该从logits中进行top-k/top-p采样
                    selected_token = next_token
                
                # 添加到生成序列
                generated_tokens.append(int(selected_token))
                
                # 检查是否遇到结束token
                if selected_token == self.meta.end_token:
                    print(f"Generated end token at step {step}")
                    break
                    
                if step % 10 == 0:
                    print(f"Generated {step} tokens so far...")
                    
            except Exception as e:
                print(f"Error during generation at step {step}: {e}")
                break
        
        print(f"Generation completed. Total tokens: {len(generated_tokens)}")
        return generated_tokens
