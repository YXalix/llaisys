#include "embedding_cpu.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
              llaisysDataType_t weight_type, size_t index_size, size_t embed_dim, size_t element_size) {
    
    // Cast index to int64_t array
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    
    // Calculate the size of one embedding vector in bytes
    size_t embed_size_bytes = embed_dim * element_size;
    
    // For each index, copy the corresponding row from weight to output
    for (size_t i = 0; i < index_size; i++) {
        int64_t row_idx = idx_ptr[i];
        
        // Calculate source and destination pointers
        const std::byte *src = weight + row_idx * embed_size_bytes;
        std::byte *dst = out + i * embed_size_bytes;
        
        // Copy the embedding vector
        std::memcpy(dst, src, embed_size_bytes);
    }
}
}
