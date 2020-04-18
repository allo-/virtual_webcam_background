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

def single_color(frame, r=255.0, g=255.0, b=255.0):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return color_filter(frame, r, g, b)

def color_filter(frame, r=255.0, g=255.0, b=255.0):
    frame[:,:,0] = frame[:,:,1] * r / 255.0
    frame[:,:,1] = frame[:,:,1] * g / 255.0
    frame[:,:,2] = frame[:,:,2] * b / 255.0
    return frame

FILTERS = {
    "blur": blur,
    "grayscale": grayscale,
    "single_color": single_color,
    "color_filter": color_filter
}

def get_filter(name):
    return FILTERS.get(name, noop)
