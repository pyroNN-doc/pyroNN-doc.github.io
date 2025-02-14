import os
from natsort import natsorted
import numpy as np

def get_images_sorted(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if "raw" in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return natsorted(images)