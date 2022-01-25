/* reference: https://github.com/NVIDIA/retinanet-examples */

#pragma once
#include <stdexcept>
#include <cstdint>
#include <thrust/functional.h>

#define CUDA_ALIGN 256

struct float6
{
  float x1, y1, x2, y2, s, c; 
};

inline __host__ __device__ float6 make_float6(float4 f, float2 t)
{
  float6 fs;
  fs.x1 = f.x; fs.y1 = f.y; fs.x2 = f.z; fs.y2 = f.w; fs.s = t.x; fs.c = t.y;
  return fs;
}

template <typename T>
inline size_t get_size_aligned(size_t num_elem) {
    size_t size = num_elem * sizeof(T);
    size_t extra_align = 0;
    if (size % CUDA_ALIGN != 0) {
        extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
    }
    return size + extra_align;
}

template <typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace, size_t &workspace_size) {
  size_t size = get_size_aligned<T>(num_elem);
  if (size > workspace_size) {
    throw std::runtime_error("Workspace is too small!");
  }
  workspace_size -= size;
  T *ptr = reinterpret_cast<T *>(workspace);
  workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
  return ptr;
}
