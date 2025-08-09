#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
              llaisysDataType_t weight_type, size_t index_size, size_t embed_dim, size_t element_size);
}
