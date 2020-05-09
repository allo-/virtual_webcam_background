import cv2
import filters
import numpy as np


class SolidColor:
    def __init__(self, r=0.0, g=0.0, b=0.0, *args, **kwargs):
        self.rgb = np.array([r, g, b])
        self.rgba = np.array([r, g, b, 255.0])

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if frame.shape[2] == 4:
            frame[:,:] = self.rgba
        elif frame.shape[2] == 3:
            frame[:,:] = self.rgb
        return frame


class Colorize:
    def __init__(self, r=255.0, g=255.0, b=255.0, *args, **kwargs):
        self.color_filter = ColorFilter(r, g, b)

    def apply(self, *args, **kwargs):
        frame = kwargs['frame'].astype(np.uint8)
        gray_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        frame[:,:,:3] = gray_frame
        kwargs['frame'] = frame.astype(np.float)
        return self.color_filter.apply(*args, **kwargs)


class ColorFilter:
    def __init__(self, r=255.0, g=255.0, b=255.0, *args, **kwargs):
        self.r = r
        self.g = g
        self.b = b

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        frame[:,:,0] = frame[:,:,0] * self.r / 255.0
        frame[:,:,1] = frame[:,:,1] * self.g / 255.0
        frame[:,:,2] = frame[:,:,2] * self.b / 255.0
        frame = np.clip(frame, 0.0, 255.0)
        return frame


filters.register_filter("solid_color", SolidColor)
filters.register_filter("colorize", Colorize)
filters.register_filter("color_filter", ColorFilter)
