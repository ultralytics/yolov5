# RoIAlign for PyTorch
This is a PyTorch version of [RoIAlign](https://arxiv.org/abs/1703.06870).
This implementation is based on `crop_and_resize`
and supports both forward and backward on CPU and GPU.

**NOTE:** Thanks [meikuam](https://github.com/meikuam) for updating 
this repo for ***PyTorch 1.0***. You can find the original version for 
`torch <= 0.4.1` in [pytorch_0.4](https://github.com/longcw/RoIAlign.pytorch/tree/pytorch_0.4)
branch.


## Introduction
The `crop_and_resize` function is ported from [tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize),
and has the same interface with tensorflow version, except the input feature map
should be in `NCHW` order in PyTorch.
They also have the same output value (error < 1e-5) for both forward and backward as we expected,
see the comparision in `test.py`.

**Note:**
Document of `crop_and_resize` can be found [here](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize).
And `RoIAlign` is a wrap of `crop_and_resize`
that uses boxes with *unnormalized `(x1, y1, x2, y2)`* as input
(while `crop_and_resize` use *normalized `(y1, x1, y2, x2)`* as input).
See more details about the difference of
 `RoIAlign` and `crop_and_resize` in [tensorpack](https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301).

**Warning:**
Currently it only works using the default GPU (index 0)

## Usage
+ Install and test
    ```
    python setup.py install
    ./test.sh
    ```

+ Use RoIAlign or crop_and_resize 
    
    Since PyTorch 1.2.0 [Legacy autograd function with non-static forward method is deprecated.](https://github.com/pytorch/pytorch/blob/fdfc676eb6c4d9f50496e564976fbe6d124e23a5/torch/csrc/autograd/python_function.cpp#L636-L638)
    We use new-style autograd function with static forward method. Example:
    ```python
    import torch
    from roi_align import RoIAlign      # RoIAlign module
    from roi_align import CropAndResize # crop_and_resize module
    
    # input feature maps (suppose that we have batch_size==2)
    image = torch.arange(0., 49).view(1, 1, 7, 7).repeat(2, 1, 1, 1)
    image[0] += 10
    print('image: ', image)
    
    
    # for example, we have two bboxes with coords xyxy (first with batch_id=0, second with batch_id=1).
    boxes = torch.Tensor([[1, 0, 5, 4],
                         [0.5, 3.5, 4, 7]])
    
    box_index = torch.tensor([0, 1], dtype=torch.int) # index of bbox in batch
    
    # RoIAlign layer with crop sizes:
    crop_height = 4
    crop_width = 4
    roi_align = RoIAlign(crop_height, crop_width)
    
    # make crops:
    crops = roi_align(image, boxes, box_index)
    
    print('crops:', crops)
    ```
    Output:
    ```python
    image:  tensor([[[[10., 11., 12., 13., 14., 15., 16.],
          [17., 18., 19., 20., 21., 22., 23.],
          [24., 25., 26., 27., 28., 29., 30.],
          [31., 32., 33., 34., 35., 36., 37.],
          [38., 39., 40., 41., 42., 43., 44.],
          [45., 46., 47., 48., 49., 50., 51.],
          [52., 53., 54., 55., 56., 57., 58.]]],


        [[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
          [ 7.,  8.,  9., 10., 11., 12., 13.],
          [14., 15., 16., 17., 18., 19., 20.],
          [21., 22., 23., 24., 25., 26., 27.],
          [28., 29., 30., 31., 32., 33., 34.],
          [35., 36., 37., 38., 39., 40., 41.],
          [42., 43., 44., 45., 46., 47., 48.]]]])
          
    crops: tensor([[[[11.0000, 12.0000, 13.0000, 14.0000],
              [18.0000, 19.0000, 20.0000, 21.0000],
              [25.0000, 26.0000, 27.0000, 28.0000],
              [32.0000, 33.0000, 34.0000, 35.0000]]],
    
    
            [[[24.5000, 25.3750, 26.2500, 27.1250],
              [30.6250, 31.5000, 32.3750, 33.2500],
              [36.7500, 37.6250, 38.5000, 39.3750],
              [ 0.0000,  0.0000,  0.0000,  0.0000]]]])
    ```
