#pragma once
#include "llaisys/models/qwen2.h"

namespace llaisys {
    struct LlaisysQwen2Model *Qwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    void Qwen2ModelDestroy(struct LlaisysQwen2Model *model);
    struct LlaisysQwen2Weights *Qwen2ModelWeights(struct LlaisysQwen2Model *model);
    int64_t Qwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken);
} // namespace llaisys