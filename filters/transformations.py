import filters
import cv2

def flip(horizontal=True, vertical=False, *args, **kwargs):
    frame = kwargs['frame']

    if horizontal:
        frame = cv2.flip(frame, 1)
    if vertical:
        frame = cv2.flip(frame, 0)

    return frame

filters.register_filter("flip", flip)
