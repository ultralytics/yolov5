#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width;
        const float top_left = pimage[top_y_index * image_width + left_x_index];
        const float top_right = pimage[top_y_index * image_width + right_x_index];
        const float bottom_left = pimage[bottom_y_index * image_width + left_x_index];
        const float bottom_right = pimage[bottom_y_index * image_width + right_x_index];

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}

__global__
void CropAndResizeBackpropImageKernel(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width;
        const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
        atomicAdd(
            pimage + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop
        );
        atomicAdd(
            pimage + top_y_index * image_width + right_x_index, 
            x_lerp * dtop
        );

        const float dbottom = y_lerp * grads_ptr[out_idx];
        atomicAdd(
            pimage + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom
        );
        atomicAdd(
            pimage + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom
        );
    }
}


void CropAndResizeLaucher(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr, cudaStream_t stream)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}