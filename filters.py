import cv2

def grayscale(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

def blur(frame, intensity):
    return cv2.blur(frame, (intensity, intensity))

