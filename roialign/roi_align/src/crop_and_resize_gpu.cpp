#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x.dim() == 4, #x " must have 4 dimensions")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Float, #x " must be float Tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Int, #x " must be int Tensor")
//using namespace at;


namespace torch {
void crop_and_resize_gpu_forward(
    torch::Tensor image,
    torch::Tensor boxes,           // [y1, x1, y2, x2]
    torch::Tensor box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    torch::Tensor crops
) {
    CHECK_INPUT(image);     CHECK_FLOAT(image);     CHECK_DIMS(image);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes);
    CHECK_INPUT(box_index); CHECK_INT(box_index);
    CHECK_INPUT(crops);     CHECK_FLOAT(crops);

    const int batch_size    = image.size(0);
    const int depth         = image.size(1);
    const int image_height  = image.size(2);
    const int image_width   = image.size(3);

    const int num_boxes     = boxes.size(0);

    // init output space
//    THCTensor_resize(state, crops, {num_boxes, depth, crop_height, crop_width});

    crops.resize_({num_boxes, depth, crop_height, crop_width});
    crops.zero_();
//    THCudaTensor_resize4d(state, crops, num_boxes, depth, crop_height, crop_width);
//    THCudaTensor_zero(state, crops);



//    auto state = globalContext().getTHCState();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();// THCState_getCurrentStream(state);

    CropAndResizeLaucher(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data<float>(),
        stream
    );
}


void crop_and_resize_gpu_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [y1, x1, y2, x2]
    torch::Tensor box_index,    // range in [0, batch_size)
    torch::Tensor grads_image // resize to [bsize, c, hc, wc]
) {
    CHECK_INPUT(grads);     CHECK_FLOAT(grads);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes);
    CHECK_INPUT(box_index); CHECK_INT(box_index);
    CHECK_INPUT(grads_image); CHECK_FLOAT(grads_image); CHECK_DIMS(grads_image);

    // shape
    const int batch_size    = grads_image.size(0);
    const int depth         = grads_image.size(1);
    const int image_height  = grads_image.size(2);
    const int image_width   = grads_image.size(3);

    const int num_boxes     = grads.size(0);
    const int crop_height   = grads.size(2);
    const int crop_width    = grads.size(3);

    // init output space
    grads_image.zero_();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CropAndResizeBackpropImageLaucher(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        grads_image.data<float>(),
        stream
    );
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &torch::crop_and_resize_gpu_forward,
      "crop_and_resize_gpu_forward");
  m.def(
      "backward",
      &torch::crop_and_resize_gpu_backward,
      "crop_and_resize_gpu_backward");
}
