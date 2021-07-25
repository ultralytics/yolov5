#ifndef _CropAndResize_Kernel
#define _CropAndResize_Kernel

//#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr, cudaStream_t stream);

void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif