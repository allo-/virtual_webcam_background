import filters
import cv2
import numpy as np


class Noise:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        noise = np.zeros((frame.shape[0], frame.shape[1], 4))
        indices = (np.random.random(frame.shape[:2]) < 0.05)
        frame[indices,0] = 255
        frame[indices,1] = 255
        frame[indices,2] = 255
        return frame

filters.register_filter("noise", Noise)
