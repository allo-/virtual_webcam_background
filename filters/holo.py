import filters
import numpy as np
import cv2

# Copied from https://elder.dev/posts/open-source-virtual-background/

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

def hologram_effect(img):
    # add a blue tint
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    holo = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)

    # add a halftone effect
    bandLength, bandGap = 2, 3

    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)

    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)

    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out


class Holo:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        data = frame[:,:,:3]
        data = hologram_effect(data)
        frame[:,:,:3] = data
        return frame

filters.register_filter("holo", Holo)
