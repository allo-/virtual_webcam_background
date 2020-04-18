import cv2

def grayscale(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

def blur(frame, intensity_x=5, intensity_y=-1):
    if intensity_x <= 0 and intensity_y <= 0:
        return frame
    if intensity_y == -1:
        intensity_y = intensity_x
    return cv2.blur(frame, (intensity_x, intensity_y))

def noop(frame, *args, **kwargs):
    return frame

FILTERS = {
    "blur": blur,
    "grayscale": grayscale
}

def get_filter(name):
    return FILTERS.get(name, noop)
