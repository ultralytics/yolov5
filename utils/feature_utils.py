import torch
from typing import Tuple


def bbox_to_pixel_map(nc: int, batch_size: int, boxes: torch.Tensor, img_shape: Tuple[int, int]):
    """
    Converts a tensor of bounding box information for a minibatch of examples into a pixel map for each class.
    If the bounding box extends anywhere into the space of a pixel, the entire pixel is marked as a '1'.

    TODO: this should be altered to take in the class confidence of each box and use that to compute the pixel map

    :param nc: the number of classes this model discriminates between
    :param batch_size: the number of images these boxes correspond to
    :param boxes: A Tensor with a row for each bounding box and a column with 6 values, in the following order:
        image_idx (which image in the minibatch the box is for), class, x, y, w, h.
        X and Y are 0 at the top left of the image.
    :param img_shape: the target output shape, a tuple of 2 integers - image width and image height.
    :return: A Tensor of pixel values between 0 and 1. Its shape will be [batch_size, nc, img_shape[0], img_shape[1]]
    """

    max_width_px = img_shape[0]
    max_height_px = img_shape[1]

    # TODO: ensure new tensors created are on the same device as 'boxes'
    pixel_bounds = torch.zeros((boxes.shape[0], 6))

    # transfer over image, class information
    pixel_bounds[:, 0:2] = boxes[:, 0:2]

    # horiz left bound
    pixel_bounds[:, 2] = torch.floor(max_width_px*(boxes[:, 2] - boxes[:, 4]/2))

    # horiz right bound
    pixel_bounds[:, 3] = torch.ceil(max_width_px*(boxes[:, 2] + boxes[:, 4]/2))

    # vert top bound
    pixel_bounds[:, 4] = torch.floor(max_height_px*(boxes[:, 3] - boxes[:, 5]/2))

    # vert bottom bound
    pixel_bounds[:, 5] = torch.ceil(max_height_px * (boxes[:, 3] + boxes[:, 5]/2))

    torch.clamp(pixel_bounds[:, 2:4], 0, max_width_px - 1)
    torch.clamp(pixel_bounds[:, 4:], 0, max_height_px - 1)
    pixel_bounds = pixel_bounds.type(torch.LongTensor)

    # TODO: ensure this new tensor is on the same device as 'boxes'
    output = torch.zeros((batch_size, nc, max_width_px, max_height_px))

    # TODO: consider a vectorized alternative
    for i in range(pixel_bounds.shape[0]):
        img_idx = pixel_bounds[i, 0]
        class_idx = pixel_bounds[i, 1]
        horiz_left_bound = pixel_bounds[i, 2]

        """
        This bound is *inclusive* but torch indexing treats it as exclusive, so we add 1.
        We do the same for 'vert_bottom_bound'.
        """
        horiz_right_bound = pixel_bounds[i, 3] + 1
        vert_top_bound = pixel_bounds[i, 4]
        vert_bottom_bound = pixel_bounds[i, 5] + 1

        output[img_idx, class_idx, horiz_left_bound:horiz_right_bound, vert_top_bound:vert_bottom_bound] = 1

    return output


if __name__ == "__main__":
    # some test code to visualize the effect
    boxes = torch.zeros((3, 6))
    boxes[0, :] = torch.Tensor([0, 0, 0.75, 0.25, 0.1, 0.3]) # a box in the top right corner.
    boxes[1, :] = torch.Tensor([0, 0, 0.2, 0.75, 0.2, 0.3]) # a box in the top left corner
    boxes[2, :] = torch.Tensor([0, 0, 0.95, 0.8, 0.05, 0.05]) # box in the bottom right corner

    pix_map_tensor = bbox_to_pixel_map(1, 1, boxes, (16, 16))
    pix_map = pix_map_tensor.cpu().detach().numpy()