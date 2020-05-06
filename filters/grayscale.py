import cv2
import numpy as np
import filters


class Grayscale:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        frame = kwargs['frame'].astype(np.uint8)
        gray_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        frame[:,:,:3] = gray_frame.astype(np.float)
        return frame


filters.register_filter("grayscale", Grayscale)
