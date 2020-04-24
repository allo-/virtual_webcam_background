import filters
import numpy as np

def change_alpha(alpha_change=0, alpha_min=0, alpha_max=255.0, *args, **kwargs):
    frame = kwargs['frame']
    if frame.shape[2] == 4:
        frame[:,:,3] = np.clip(frame[:,:,3].astype(np.int16) + alpha_change,
            alpha_min, alpha_max)
    elif frame.shape[2] == 3:
        frame = np.clip(np.append(
            frame.astype(np.int16),
            np.ones((frame.shape[0], frame.shape[1], 1)) * 255 + alpha_change,
            axis=2),
        alpha_min, alpha_max)
    return frame

filters.register_filter("change_alpha", change_alpha)
