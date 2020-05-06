import cv2
import filters
import numpy as np

def single_color(r=255.0, g=255.0, b=255.0, *args, **kwargs):
    frame = kwargs['frame'].astype(np.uint8)
    gray_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    frame[:,:,:3] = gray_frame
    kwargs['frame'] = frame.astype(np.float)
    return color_filter(r=r, g=g, b=b, *args, **kwargs)

def color_filter(r=255.0, g=255.0, b=255.0, *args, **kwargs):
    frame = kwargs['frame']
    frame[:,:,0] = frame[:,:,0] * r / 255.0
    frame[:,:,1] = frame[:,:,1] * g / 255.0
    frame[:,:,2] = frame[:,:,2] * b / 255.0
    frame = np.clip(frame, 0.0, 255.0)
    return frame

filters.register_filter("single_color", single_color)
filters.register_filter("color_filter", color_filter)
