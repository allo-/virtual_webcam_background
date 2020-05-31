import filters
import cv2
import numpy as np
from scipy import ndimage


class Anonymize:
    def __init__(self, blur=20, padding=10, secure=False, *args, **kwargs):
        self.padding = padding
        self.blur = blur
        self.secure = secure

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        part_masks = kwargs['part_masks']

        face_mask = np.bitwise_or(part_masks[:,:,0], part_masks[:,:,1])
        objs = ndimage.find_objects(face_mask)
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        for obj in objs:
            min_x, min_y = min(min_x, obj[0].start), min(min_y, obj[1].start)
            max_x, max_y = max(max_x, obj[0].stop), max(max_y, obj[1].stop)

        min_x = max(0, min_x - self.padding)
        min_y = max(0, min_y - self.padding)
        max_x = min(frame.shape[0], max_x + self.padding)
        max_y = min(frame.shape[1], max_y + self.padding)

        if np.isfinite([min_x, max_x, min_y, max_y]).all():
            face_mask[min_x:max_x,min_y:max_y] = 1.0
        elif self.secure:
            # When no face is detected, anonymize everything
            face_mask[:,:] = 1.0

        face_mask = np.expand_dims(face_mask, axis=2)
        if self.blur:
            anonymized_frame = cv2.blur(frame, (self.blur, self.blur))
        else:
            anonymized_frame = frame
            anonymized_frame[:,:,:3] = 0.0
        anonymized_frame[:,:,:4] = anonymized_frame[:,:,:4] * face_mask

        return anonymized_frame


filters.register_filter("anonymize", Anonymize)
