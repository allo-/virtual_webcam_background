import filters
import cv2
import numpy as np
from scipy import ndimage


class Halo:
    def __init__(self, zoom=1.0, distance=0.0, *args, **kwargs):
        self.zoom = zoom
        self.distance = distance

        self.halo = cv2.imread("images/halo.png", cv2.IMREAD_UNCHANGED)
        self.halo = cv2.resize(self.halo,
            dsize=(int(self.halo.shape[1] * zoom), int(self.halo.shape[0] * zoom)),
            interpolation=cv2.INTER_AREA)
        # BGR to RGB
        self.halo[:,:,0], self.halo[:,:,2] = \
                self.halo[:,:,2], self.halo[:,:,0].copy()
        self.center_of_halo = \
                np.array([self.halo.shape[0] / 2, self.halo.shape[1] / 2])

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        part_masks = kwargs['part_masks']
        halo = self.halo

        face_mask = np.bitwise_or(part_masks[:,:,0], part_masks[:,:,1])

        center_of_mass = np.array(ndimage.center_of_mass(face_mask))
        mask_top = ndimage.find_objects(face_mask)[0][0].start

        mask_top -= self.distance


        halo_top = int(mask_top - halo.shape[0])
        halo_bottom = halo_top + halo.shape[0]
        halo_left = int(center_of_mass[1] - self.center_of_halo[1])
        halo_right = halo_left + halo.shape[1]

        cropped_halo = halo[
            -min(0, halo_top):
            halo.shape[0] + min(0, frame.shape[0] - halo_bottom),
            -min(0, halo_left):
            halo.shape[1] + min(0, frame.shape[1] - halo_right)
        ]

        halo_top = max(0, mask_top - cropped_halo.shape[0])
        halo_bottom = halo_top + cropped_halo.shape[0]
        halo_left = max(0, halo_left)
        halo_right = halo_left + cropped_halo.shape[1]

        alpha = cropped_halo[:,:,3]
        for c in range(3):
            frame[halo_top:halo_bottom,halo_left:halo_right,c] = \
                frame[halo_top:halo_bottom,halo_left:halo_right,c] \
                * (1.0 - alpha / 255.0) + cropped_halo[:,:,c] * (alpha / 255.0)

        return frame


filters.register_filter("halo", Halo)
