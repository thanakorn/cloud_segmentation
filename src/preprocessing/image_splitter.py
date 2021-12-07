import numpy as np
from itertools import product

class ImageSplitter:
    @classmethod
    def split_image(cls, img, patch_height, patch_width):
        height, width = img.shape[0], img.shape[1]
        offsets = product(range(0, height, patch_height), range(0, width, patch_width))
        patches = [ img[row_offset:(row_offset + patch_height), col_offset:(col_offset + patch_width)]
                    for row_offset, col_offset in offsets ]
        return patches