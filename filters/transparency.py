import filters
import numpy as np


class ChangeAlpha:
    def __init__(self,
                 alpha_change=0,
                 alpha_min=0,
                 alpha_max=255.0,
                 *args, **kwargs):
        self.alpha_change = alpha_change
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if frame.shape[2] == 4:
            frame[:,:,3] = np.clip(frame[:,:,3] + self.alpha_change,
                                   self.alpha_min, self.alpha_max)
        elif frame.shape[2] == 3:
            frame = np.clip(np.append(
                frame,
                np.ones((frame.shape[0], frame.shape[1], 1)) * 255 + \
                        self.alpha_change, axis=2),
                self.alpha_min, self.alpha_max)
        return frame


class ChromaKey:
    def __init__(self, r=0.0, g=255.0, b=0.0, fuzz=10.0, *args, **kwargs):
        self.rgb_from = np.clip(
                np.array([r - fuzz, g - fuzz, b - fuzz]), 0, 255)
        self.rgb_to = np.clip(
                np.array([r + fuzz, g + fuzz, b + fuzz]), 0, 255)

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if frame.shape[2] == 3:
            # Add alpha channel
            frame = np.append(frame,
                    np.ones((frame.shape[0], frame.shape[1], 1)) * 255.0, axis=2)

        frame[np.min(
            (frame[:,:,:3] >= self.rgb_from) & (frame[:,:,:3] <= self.rgb_to),
            axis = 2), 3] = 0
        return frame


filters.register_filter("change_alpha", ChangeAlpha)
filters.register_filter("chroma_key", ChromaKey)
