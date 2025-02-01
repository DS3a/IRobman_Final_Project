import numpy as np
import cv2

def segmentation_mask(seg_image, id=5):
    """
        This creates a binary mask containing only the id given, and with everything else set to 0

        This mask can then be applied to whichever image of the same width for further processing
    """

    mask = np.where(seg_image == id, seg_image, 0)
    mask = 255*(mask / 5)
    return mask.astype(np.uint8)

def segment_depth_image(depth_image, mask):
    # just to verify that they are compatible
    assert depth_image.shape == mask.shape
    return cv2.bitwise_and(depth_image, depth_image, mask=mask)