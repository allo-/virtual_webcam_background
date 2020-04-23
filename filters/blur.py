import cv2
import filters

def blur(intensity_x=5, intensity_y=-1, *args, **kwargs):
    frame = kwargs['frame']
    if intensity_x <= 0 and intensity_y <= 0:
        return frame
    if intensity_y == -1:
        intensity_y = intensity_x
    return cv2.blur(frame, (intensity_x, intensity_y))

filters.register_filter("blur", blur)
