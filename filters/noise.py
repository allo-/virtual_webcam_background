import filters
import numpy as np


class Noise:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        indices = (np.random.random(frame.shape[:2]) < 0.05)
        frame[indices,:] = 255
        return frame

filters.register_filter("noise", Noise)
