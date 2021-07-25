import numpy as np
import torch
import sys
from torch import nn
from torch.autograd import Variable, gradcheck
try:
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
except:
    print("Unexpected error:", sys.exc_info()[0])
    tf = None

from roi_align.crop_and_resize import CropAndResizeFunction
from roi_align.roi_align import RoIAlign


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def generate_data(batch_size, depth, im_height, im_width, n_boxes, xyxy=False, box_normalize=True):

    # random rois
    xs = np.random.uniform(0, im_width, size=(n_boxes, 2))
    ys = np.random.uniform(0, im_height, size=(n_boxes, 2))
    if box_normalize:
        xs /= (im_width - 1)
        ys /= (im_height - 1)

    xs.sort(axis=1)
    ys.sort(axis=1)

    if xyxy:
        boxes_data = np.stack((xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]), axis=-1).astype(np.float32)
    else:
        boxes_data = np.stack((ys[:, 0], xs[:, 0], ys[:, 1], xs[:, 1]), axis=-1).astype(np.float32)
    box_index_data = np.random.randint(0, batch_size, size=n_boxes, dtype=np.int32)
    image_data = np.random.randn(batch_size, depth, im_height, im_width).astype(np.float32)

    return image_data, boxes_data, box_index_data


def compare_with_tf(crop_height, crop_width, is_cuda=True):
    # generate data
    image_data, boxes_data, box_index_data = generate_data(
        batch_size=2,
        depth=128,
        im_height=200,
        im_width=200,
        n_boxes=10,
        xyxy=False, box_normalize=True)
    # boxes_tf_data = np.stack((boxes_data[:, 1], boxes_data[:, 0], boxes_data[:, 3], boxes_data[:, 2]), axis=1)
    # boxes_tf_data[:, 0::2] /= (image_data.shape[2] - 1.)
    # boxes_tf_data[:, 1::2] /= (image_data.shape[3] - 1.)

    # rand conv layer
    conv_torch = nn.Conv2d(image_data.shape[1], 64, 3, padding=1, bias=False)
    if is_cuda:
        conv_torch = conv_torch.cuda()

    # pytorch forward
    image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
    boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

    print('pytorch forward and backward start')
    crops_torch = CropAndResizeFunction.apply(image_torch, boxes, box_index, crop_height, crop_width, 0)
    crops_torch = conv_torch(crops_torch)
    crops_torch_data = crops_torch.data.cpu().numpy()

    # pytorch backward
    loss_torch = crops_torch.sum()
    loss_torch.backward()
    grad_torch_data = image_torch.grad.data.cpu().numpy()

    print('pytorch forward and backward end')

    # tf forward & backward
    image_tf = tf.placeholder(tf.float32, (None, None, None, None), name='image')
    boxes = tf.placeholder(tf.float32, (None, 4), name='boxes')
    box_index = tf.placeholder(tf.int32, (None,), name='box_index')

    image_t = tf.transpose(image_tf, (0, 2, 3, 1))
    crops_tf = tf.image.crop_and_resize(image_t, boxes, box_index, (crop_height, crop_width))
    conv_tf = tf.nn.conv2d(crops_tf, np.transpose(conv_torch.weight.data.cpu().numpy(), (2, 3, 1, 0)),
                           [1, 1, 1, 1], padding='SAME')

    trans_tf = tf.transpose(conv_tf, (0, 3, 1, 2))
    loss_tf = tf.reduce_sum(trans_tf)
    grad_tf = tf.gradients(loss_tf, image_tf)[0]

    with tf.Session() as sess:
        crops_tf_data, grad_tf_data = sess.run(
            (trans_tf, grad_tf), feed_dict={image_tf: image_data, boxes: boxes_data, box_index: box_index_data}
        )

    crops_diff = np.abs(crops_tf_data - crops_torch_data)
    print('forward (maxval, min_err, max_err, mean_err):',
          crops_tf_data.max(), crops_diff.min(), crops_diff.max(), crops_diff.mean())

    grad_diff = np.abs(grad_tf_data - grad_torch_data)
    print('backward (maxval, min_err, max_err, mean_err):',
          grad_tf_data.max(), grad_diff.min(), grad_diff.max(), grad_diff.mean())


def test_roialign(is_cuda=True):
    # generate data
    crop_height = 3
    crop_width = 3
    image_data, boxes_data, box_index_data = generate_data(
        batch_size=2,
        depth=2,
        im_height=10,
        im_width=10,
        n_boxes=2,
        xyxy=True, box_normalize=False)
    max_inp = np.abs(image_data).max()
    print('max_input:', max_inp)

    image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
    boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

    roi_align = RoIAlign(crop_height, crop_width, transform_fpcoor=False)
    gradcheck(roi_align, (image_torch, boxes, box_index), eps=max_inp/500)

    print('test ok')


if __name__ == '__main__':
    def main():
        crop_height = 7
        crop_width = 7
        is_cuda = torch.cuda.is_available()

        if tf is not None:
            compare_with_tf(crop_height, crop_width, is_cuda=is_cuda)
        else:
            print('without tensorflow')

        test_roialign(is_cuda=is_cuda)

    main()
