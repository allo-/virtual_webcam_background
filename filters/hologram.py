import cv2
import filters
import numpy as np

# Based on the blog post of Benjamin Elder
# https://elder.dev/posts/open-source-virtual-background/
# License: CC-BY 4.0

def shift_image(input_img, dx, dy):
    img = np.roll(input_img, dy, axis=0)
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

class Hologram:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        frame = kwargs['frame'].astype(np.uint8)
        frame[:,:,:3] = cv2.applyColorMap(frame[:,:,:3], cv2.COLORMAP_WINTER)
        frame[:,:,:3] = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2RGB)
        # add a halftone effect
        bandLength, bandGap = 2, 3
        for y in range(frame.shape[0]):
            if y % (bandLength+bandGap) < bandLength:
                frame[y,:,:3] = frame[y,:,:3] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(frame[:,:,:3], 0.2,
                                    shift_image(frame[:,:,:3], 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4,
                                    shift_image(frame[:,:,:3], -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        frame[:,:,:3] = cv2.addWeighted(frame[:,:,:3], 0.5, holo_blur, 0.6, 0)
        return frame.astype(np.float)


filters.register_filter("hologram", Hologram)
