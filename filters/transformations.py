import filters
import cv2


class Flip:
    def __init__(self, horizontal=True, vertical=False, *args, **kwargs):
        self.horizontal = horizontal
        self.vertical = vertical

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if self.horizontal:
            frame = cv2.flip(frame, 1)
        if self.vertical:
            frame = cv2.flip(frame, 0)
        return frame


filters.register_filter("flip", Flip)
