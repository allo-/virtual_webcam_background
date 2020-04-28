import cv2
import filters

def blur(intensity_x=5, intensity_y=-1, *args, **kwargs):
    frame = kwargs['frame']
    if intensity_x <= 0 and intensity_y <= 0:
        return frame
    if intensity_y < 0:
        intensity_y = intensity_x
    # Do not blur the alpha channel
    frame[:,:,:3] = cv2.blur(frame[:,:,:3], (intensity_x, intensity_y))
    return frame

filters.register_filter("blur", blur)
