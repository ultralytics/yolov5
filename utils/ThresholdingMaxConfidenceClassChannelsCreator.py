import logging
import torch
import ClassChannelsCreator
from typing import Tuple

class ThresholdingMaxConfidenceClassChannelsCreator(ClassChannelsCreator.ClassChannelsCreator):
    def  __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        assert threshold >= 0 and threshold <= 1, "threshold should be between 0 and 1"
        self.threshold = threshold 

    def predicted_bboxes_to_pixel_map(self, boxes: torch.Tensor, img_shape: Tuple[int, int],
                                      keep_top_n_boxes: int = ClassChannelsCreator.DEFAULT_BOX_LIMIT) -> torch.Tensor:
        """
        Converts a tensor of bounding box information for a minibatch of examples into a pixel map for each class.
        Each class' pixel map has the highest confidence value of any box containing that pixel. The confidence value is
        the product of the objectness score and the class confidence. The 'boxes' tensor may be obtained from the output
        of a yolo.Model's 'forward()' method in inference mode.

        :param boxes: a Tensor of box predictions organized like [batch_size, num_boxes, features]
            'features' has the following values, in this order: x, y, w, h, objectness, class_scores
            where class_scores is num_classes long, from 0 to the max class index.
        :param img_shape: the target output shape, a tuple of 2 integers - image width and image height.
        :param keep_top_n_boxes: An integer that controls the number of boxes used for the pixel map.
            Higher values are more accurate but slower.
        :return: A Tensor of pixel values between 0 and 1. Its shape will be [batch_size, nc, img_shape[0], img_shape[1]]
        """
        logger = logging.getLogger(__name__)

        batch_size, num_boxes, numChannels = boxes.shape
        numChannels -= 5  # there are five more entries than the number of classes in the last dim of 'predictions'

        max_width_px = img_shape[0]
        max_height_px = img_shape[1]

        # Get indices into the box matrix where objectness score is greater than threshold
        boxesIndicesObjectnessGreaterThanThreshold = boxes[:,:,4] >= self.threshold 

        # Pixel bounds contains clamped pixel values where shape is (BatchSize, #Boxes, 7) where 7 is class_conf, objness, hleft, hright, vtop, vbottom, class_idx
        pixel_bounds = torch.zeros((boxes.shape[0], boxes.shape[1], 7)).to(boxes.device)

        # transfer over class, objectness information
        pixel_bounds[:, :, 0], pixel_bounds[:, :, 6] = torch.max(boxes[:, :, 5:], dim=2)
        # The above statement is a more efficient way to do:
        # pixel_bounds[:, :, 6] = torch.argmax(boxes[:, :, 5:], dim=2)
        # pixel_bounds[:, :, 0], _ = torch.max(boxes[:, :, 5:], dim=2)

        pixel_bounds[:, :, 1] = boxes[:, :, 4]  # objectness

        # the following pixel boundaries are inclusive
        # horiz left bound = centerX - width/2
        pixel_bounds[:, :, 2] = torch.floor(max_width_px*(boxes[:, :, 0] - boxes[:, :, 2]/2))

        # horiz right bound = centerX + width/2
        pixel_bounds[:, :, 3] = torch.ceil(max_width_px*(boxes[:, :, 0] + boxes[:, :, 2]/2))

        # vert top bound = centerY - height/2
        pixel_bounds[:, :, 4] = torch.floor(max_height_px*(boxes[:, :, 1] - boxes[:, :, 3]/2))

        # vert bottom bound = centerY + height/2
        pixel_bounds[:, :, 5] = torch.ceil(max_height_px * (boxes[:, :, 1] + boxes[:, :, 3]/2))

        # Make sure values are between 0 and width of image
        torch.clamp(pixel_bounds[:, :, 2:4], 0, max_width_px - 1)

        # Make sure values are between 0 and height of image
        torch.clamp(pixel_bounds[:, :, 4:6], 0, max_height_px - 1)
        pixel_bounds_int_parts = pixel_bounds[:, :, 2:].clone().to(boxes.device).long()  # TODO: keep int_parts in separate tensor the whole time?

        output = torch.zeros((batch_size, numChannels, max_width_px, max_height_px)).to(boxes.device)

        numImagesInBatch = pixel_bounds.shape[0] 
        numBoxes = pixel_bounds.shape[1]
        # Iterate over (#images in batch, #boxes) and updates the output. 
        countBoxesWhoseObjectnessIsGreaterThanThreshold: int = 0
        for imageIndex in range(numImagesInBatch):
            for boxIndex in range(numBoxes):
                if boxesIndicesObjectnessGreaterThanThreshold[imageIndex, boxIndex] == False:
                    continue

                countBoxesWhoseObjectnessIsGreaterThanThreshold += 1
                class_idx = pixel_bounds_int_parts[imageIndex, boxIndex, 4]
                horiz_left_bound = pixel_bounds_int_parts[imageIndex, boxIndex, 0]
                vert_top_bound = pixel_bounds_int_parts[imageIndex, boxIndex, 2]
                """
                This bound is *inclusive* but torch indexing treats it as exclusive, so we add 1.
                We do the same for 'vert_bottom_bound'.
                """
                horiz_right_bound = pixel_bounds_int_parts[imageIndex, boxIndex, 1] + 1
                vert_bottom_bound = pixel_bounds_int_parts[imageIndex, boxIndex, 3] + 1

                confidence = pixel_bounds[imageIndex, boxIndex, 0] # Keep the highest score

                output[imageIndex, class_idx, horiz_left_bound:horiz_right_bound, vert_top_bound:vert_bottom_bound] = torch.max(output[imageIndex, class_idx, horiz_left_bound:horiz_right_bound, vert_top_bound:vert_bottom_bound], confidence)

        logger.debug("CountBoxesWhoseObjectnessIsGreaterThanThreshold %i", countBoxesWhoseObjectnessIsGreaterThanThreshold)
        return output

def test_pixel_map():
    boxes = torch.rand(size=(1, 4, 7))*(2/3)
    boxes[0, 0, :] = torch.Tensor([0.5, 0.5, 0.5, 0.5, 1, 1, 0])  # class 0 has a square in the center of its map
    boxes[0, 1, :] = torch.Tensor([0.5, 0.5, 0.5, 0.5, 1, 0, 1])  # class 1 has a square in the center of its map
    boxes[0, 2, :] = torch.Tensor([0.25, 0.25, 0.5, 0.5, 1, 0.8, 0])  # class 0 has a square in the upper-left of its map
    boxes[0, 3, :] = torch.Tensor([0.75, 0.75, 0.5, 0.5, 1, 0, 1])  # class 1 has a square in the bottom-right of its map
    classChannelsCreator: ClassChannelsCreator = ThresholdingMaxConfidenceClassChannelsCreator(0.5)
    outputs = classChannelsCreator.predicted_bboxes_to_pixel_map(boxes, (8, 8))
    return boxes, outputs

if __name__ == "__main__":
    # some test code to demonstrate the effect (you can visualize it with a debugger)
    test_pixel_map()
