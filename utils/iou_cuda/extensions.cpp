/* reference: https://github.com/NVIDIA/retinanet-examples */

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cmath>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <optional>

#include "inter_union_cuda.h"
#include <stdio.h>

using namespace std;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/* Below functions are used for iou computation (polygon);
Boxes have shape nx8 and Anchors have mx8;
Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
*/
vector<at::Tensor> inter_union_cuda(at::Tensor boxes, at::Tensor anchors) {

    CHECK_INPUT(boxes);
    CHECK_INPUT(anchors);

    int num_boxes = boxes.numel() / 8;
    int num_anchors = anchors.numel() / 8;
    auto options = boxes.options();

    auto inters = at::zeros({num_boxes*num_anchors}, options);
    auto unions = at::zeros({num_boxes*num_anchors}, options);

    // Calculate Polygon IOU
    vector<void *> inputs = {boxes.data_ptr(), anchors.data_ptr()};
    vector<void *> outputs = {inters.data_ptr(), unions.data_ptr()};

    inter_union(inputs.data(), outputs.data(), num_boxes, num_anchors, at::cuda::getCurrentCUDAStream());


    auto shape = std::vector<int64_t>{num_anchors, num_boxes};

    return {inters.reshape(shape), unions.reshape(shape)};
}


/* Below functions are used for loss computation (polygon);
For boxes and anchors having the same shape: nx8;
Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
*/
vector<at::Tensor> b_inter_union_cuda(at::Tensor boxes, at::Tensor anchors) {

    CHECK_INPUT(boxes);
    CHECK_INPUT(anchors);

    int num_anchors = anchors.numel() / 8;
    auto options = boxes.options();

    auto inters = at::zeros({num_anchors}, options);
    auto unions = at::zeros({num_anchors}, options);

    // Calculate Polygon IOU
    vector<void *> inputs = {boxes.data_ptr(), anchors.data_ptr()};
    vector<void *> outputs = {inters.data_ptr(), unions.data_ptr()};

    b_inter_union(inputs.data(), outputs.data(), num_anchors, at::cuda::getCurrentCUDAStream());


    auto shape = std::vector<int64_t>{num_anchors};

    return {inters.reshape(shape), unions.reshape(shape)};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polygon_inter_union_cuda", &inter_union_cuda);
    m.def("polygon_b_inter_union_cuda", &b_inter_union_cuda);
}
