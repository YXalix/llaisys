
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t
from ctypes import (
    POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int, c_float, c_int64, c_double, Structure
)


# Qwen2 Meta结构体
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

    def __str__(self):
        return f"LlaisysQwen2Meta(dtype={self.dtype}, nlayer={self.nlayer}, hs={self.hs}, nh={self.nh}, nkvh={self.nkvh}, dh={self.dh}, di={self.di}, maxseq={self.maxseq}, voc={self.voc}, epsilon={self.epsilon}, theta={self.theta}, end_token={self.end_token})"

class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

    # struct LlaisysQwen2Model {
    #     const LlaisysQwen2Meta *meta;
    #     llaisysDeviceType_t device;
    #     int *device_ids;
    #     int ndevice;
    #     LlaisysQwen2Weights weights;
    # };

class LlaisysQwen2Model(Structure):
    _fields_ = [
        ("meta", POINTER(LlaisysQwen2Meta)),
        ("device", llaisysDeviceType_t),
        ("device_ids", POINTER(c_int)),
        ("ndevice", c_size_t),
        ("weights", LlaisysQwen2Weights),
    ]

    def __str__(self):
        return f"LlaisysQwen2Model(device={self.device}, ndevice={self.ndevice}, device_ids={self.device_ids}, meta={self.meta}, weights={self.weights})"

# Qwen2 Model/Weights句柄
LlaisysQwen2Model_p = POINTER(LlaisysQwen2Model)
LlaisysQwen2Meta_p = POINTER(LlaisysQwen2Meta)
LlaisysQwen2Weights_p = POINTER(LlaisysQwen2Weights)

def load_models(lib):
    # Qwen2 C API
    lib.llaisysQwen2ModelCreate.argtypes = [LlaisysQwen2Meta_p, llaisysDeviceType_t, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights_p

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64


