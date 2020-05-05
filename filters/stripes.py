import filters
import numpy as np

roll_y = 0

def stripes(width=5, intensity=10.0, speed=1, *args, **kwargs):
    global roll_y
    roll_y = (roll_y + speed) % (width*2)
    frame = kwargs['frame'].astype(np.float)
    for i in range(width):
        frame[i+roll_y+0::2*width,:,:3] -= intensity
        frame[i+roll_y+width::2*width,:,:3] += intensity
    return np.clip(frame, 0.0, 255.0)

filters.register_filter("stripes", stripes)
