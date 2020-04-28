import cv2
import filters

def grayscale(*args, **kwargs):
    frame = kwargs['frame']
    gray_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    frame[:,:,:3] = gray_frame
    return frame

filters.register_filter("grayscale", grayscale)
