#include "llaisys/models/qwen2.h"
#include "../../models/qwen2.hpp"

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        return llaisys::Qwen2ModelCreate(meta, device, device_ids, ndevice);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        llaisys::Qwen2ModelDestroy(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return llaisys::Qwen2ModelWeights(model);
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        return llaisys::Qwen2ModelInfer(model, token_ids, ntoken);
    }
}