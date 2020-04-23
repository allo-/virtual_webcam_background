import cv2
import filters

def grayscale(*args, **kwargs):
    frame = kwargs['frame']
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

filters.register_filter("grayscale", grayscale)
