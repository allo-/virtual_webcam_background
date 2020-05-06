import cv2
import filters


class GaussianBlur:
    def __init__(self, intensity_x=5, intensity_y=-1, *args, **kwargs):
        if intensity_y < 0:
            intensity_y = intensity_x
        if (intensity_x % 2) == 0:
            intensity_x += 1
        if (intensity_y % 2) == 0:
            intensity_y += 1

        self.intensity_x = intensity_x
        self.intensity_y = intensity_y

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if self.intensity_x <= 0 and self.intensity_y <= 0:
            return frame

        frame[:,:,:3] = cv2.GaussianBlur(frame[:,:,:3],
                (self.intensity_x, self.intensity_y), 0)
        return frame


filters.register_filter("gaussian_blur", GaussianBlur)
