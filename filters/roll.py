import filters
import numpy as np

position_x = 0
position_y = 0

def roll(speed_x=10, speed_y=0, *args, **kwargs):
    global position_x, position_y
    frame = kwargs['frame']
    position_x = (position_x + speed_x) % frame.shape[1]
    position_y = (position_y + speed_y) % frame.shape[0]
    return np.roll(frame,
        shift=(position_x, position_y), axis=(1, 0))

filters.register_filter("roll", roll)
