import cv2
import filters

def gaussian_blur(intensity_x=5, intensity_y=-1, *args, **kwargs):
    frame = kwargs['frame']
    if intensity_x <= 0 and intensity_y <= 0:
        return frame
    if intensity_y < 0:
        intensity_y = intensity_x
    if (intensity_x % 2) == 0:
        intensity_x += 1
    if (intensity_y % 2) == 0:
        intensity_y += 1

    return cv2.GaussianBlur(frame, (intensity_x, intensity_y), 0)

filters.register_filter("gaussian_blur", gaussian_blur)
