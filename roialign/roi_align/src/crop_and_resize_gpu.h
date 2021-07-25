namespace torch {
//cuda tensors
void crop_and_resize_gpu_forward(
    torch::Tensor image,
    torch::Tensor boxes,           // [y1, x1, y2, x2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    torch::Tensor crops
);

void crop_and_resize_gpu_backward(
    torch::Tensor grads,
    torch::Tensor boxes,      // [y1, x1, y2, x2]
    torch::Tensor box_index,    // range in [0, batch_size) // int
    torch::Tensor grads_image // resize to [bsize, c, hc, wc]
);
}