import filters
import cv2
import numpy as np
from scipy import ndimage


class Flip:
    def __init__(self, horizontal=True, vertical=False, *args, **kwargs):
        self.horizontal = horizontal
        self.vertical = vertical

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if self.horizontal:
            frame = cv2.flip(frame, 1)
        if self.vertical:
            frame = cv2.flip(frame, 0)
        return frame

class Zoom:
    def __init__(self, horizontal, vertical=None, pad_and_crop=True,
                 *args, **kwargs):
        self.horizontal = horizontal
        if vertical is None:
            vertical = horizontal
        self.vertical = vertical
        self.pad_and_crop = pad_and_crop

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        zoomed = ndimage.zoom(frame, (self.vertical, self.horizontal, 1.0),
                              order=0)

        if zoomed.shape[2] == 3:
            # Add alpha channel
            zoomed = np.append(zoomed,
                    np.ones((zoomed.shape[0], zoomed.shape[1], 1)) * 255.0,
                    axis=2)


        if self.pad_and_crop:
            frame = np.zeros((frame.shape[0], frame.shape[1], 4))
            frame[:min(frame.shape[0], zoomed.shape[0]),
                  :min(frame.shape[1], zoomed.shape[1]),
                  :zoomed.shape[2]] = \
            zoomed[:min(frame.shape[0], zoomed.shape[0]),
                  :min(frame.shape[1], zoomed.shape[1]),
                  :zoomed.shape[2]]
        else:
            frame = zoomed

        return frame


class Move:
    def __init__(self, horizontal, vertical, relative=False, periodic=True,
                 *args, **kwargs):

        self.horizontal = horizontal
        self.vertical = vertical
        self.relative = relative
        self.periodic = periodic

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        horizontal = self.horizontal
        vertical = self.vertical
        if self.relative:
            horizontal = int(frame.shape[1] * horizontal)
            vertical =   int(frame.shape[0] * vertical)

        if self.periodic:
            return np.roll(frame,
                           shift=(horizontal, vertical),
                           axis=(1, 0))
        else:
            return ndimage.affine_transform(frame,
                           matrix=[1, 1, 1],
                           offset=[-vertical, -horizontal, 0],
                           order=0)


filters.register_filter("flip", Flip)
filters.register_filter("zoom", Zoom)
filters.register_filter("move", Move)
