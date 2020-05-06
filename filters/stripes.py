import filters
import numpy as np


class Stripes:
    def __init__(self, width=5, intensity=10.0, speed=1, *args, **kwargs):
        self.width = width
        self.intensity = intensity
        self.speed = speed
        self.roll_y = 0

    def apply(self, *args, **kwargs):
        self.roll_y = (self.roll_y + self.speed) % (self.width * 2)
        frame = kwargs['frame']
        for i in range(self.width):
            frame[i + self.roll_y + 0::2 * self.width,:,:3] -= \
                self.intensity
            frame[i + self.roll_y + self.width::2 * self.width,:,:3] += \
                self.intensity
        return np.clip(frame, 0.0, 255.0)


filters.register_filter("stripes", Stripes)
