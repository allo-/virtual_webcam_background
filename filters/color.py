import cv2
import filters

def single_color(r=255.0, g=255.0, b=255.0, *args, **kwargs):
    frame = kwargs['frame']
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    kwargs['frame'] = frame
    return color_filter(r=r, g=g, b=b, *args, **kwargs)

def color_filter(r=255.0, g=255.0, b=255.0, *args, **kwargs):
    frame = kwargs['frame']
    frame[:,:,0] = frame[:,:,0] * r / 255.0
    frame[:,:,1] = frame[:,:,1] * g / 255.0
    frame[:,:,2] = frame[:,:,2] * b / 255.0
    return frame

filters.register_filter("single_color", single_color)
filters.register_filter("color_filter", color_filter)
