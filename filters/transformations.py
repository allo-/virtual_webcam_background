import filters
import cv2
import numpy as np
from scipy import ndimage


class Flip:
    def __init__(self, horizontal=True, vertical=False, *args, **kwargs):
        self.horizontal = horizontal
        self.vertical = vertical

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']
        if self.horizontal:
            frame = cv2.flip(frame, 1)
        if self.vertical:
            frame = cv2.flip(frame, 0)
        return frame

class Zoom:
    def __init__(self, horizontal, vertical=None, pad_and_crop=True,
                 *args, **kwargs):
        self.horizontal = horizontal
        if vertical is None:
            vertical = horizontal
        self.vertical = vertical
        self.pad_and_crop = pad_and_crop

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        zoomed = ndimage.zoom(frame, (self.vertical, self.horizontal, 1.0),
                              order=0)

        if zoomed.shape[2] == 3:
            # Add alpha channel
            zoomed = np.append(zoomed,
                    np.ones((zoomed.shape[0], zoomed.shape[1], 1)) * 255.0,
                    axis=2)


        if self.pad_and_crop:
            frame = np.zeros((frame.shape[0], frame.shape[1], 4))
            frame[:min(frame.shape[0], zoomed.shape[0]),
                  :min(frame.shape[1], zoomed.shape[1]),
                  :zoomed.shape[2]] = \
            zoomed[:min(frame.shape[0], zoomed.shape[0]),
                  :min(frame.shape[1], zoomed.shape[1]),
                  :zoomed.shape[2]]
        else:
            frame = zoomed

        return frame


class Move:
    def __init__(self, horizontal, vertical, relative=False, periodic=True,
                 *args, **kwargs):

        self.horizontal = horizontal
        self.vertical = vertical
        self.relative = relative
        self.periodic = periodic

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        horizontal = self.horizontal
        vertical = self.vertical
        if self.relative:
            horizontal = int(frame.shape[1] * horizontal)
            vertical =   int(frame.shape[0] * vertical)

        if self.periodic:
            return np.roll(frame,
                           shift=(horizontal, vertical),
                           axis=(1, 0))
        else:
            return ndimage.affine_transform(frame,
                           matrix=[1, 1, 1],
                           offset=[-vertical, -horizontal, 0],
                           order=0)

class Translate_to_head:
    @classmethod
    def config(cls):
        return {
            "Anchor Point": {"type": "enum", "options": ["HEADS", "EYES"]},
            "Average Frames": {"type": "integer", "range": [1, 100], "default": 5},
            }
    def __init__(self, anchor_point="HEADS", average_frames = 5, *args, **kwargs):
        self.anchor_point = anchor_point
        self.average_frames = average_frames

        self._avg_points = []
        self._avg_points_idx = 0

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        part_masks = kwargs['part_masks']
        heatmap_masks = kwargs['heatmap_masks']

        if self.anchor_point == "EYES":
            # left and right eye
            face_mask = np.bitwise_or(heatmap_masks[:,:,1],
                                      heatmap_masks[:,:,2])
        else:
            # left and right half of the face
            face_mask = np.bitwise_or(part_masks[:,:,0], part_masks[:,:,1])

        objs = ndimage.find_objects(face_mask)
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        for obj in objs:
            min_x, min_y = min(min_x, obj[0].start), min(min_y, obj[1].start)
            max_x, max_y = max(max_x, obj[0].stop), max(max_y, obj[1].stop)

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[0], max_x)
        max_y = min(frame.shape[1], max_y)

        if np.isfinite([min_x, max_x, min_y, max_y]).all():
            vertical = (max_x - min_x) / 2 + min_x
            horizontal = (max_y - min_y) / 2 + min_y

            if len(self._avg_points) <= self._avg_points_idx:
                self._avg_points.extend([(vertical, horizontal)])
            else:
                self._avg_points[self._avg_points_idx] = (vertical, horizontal)
            self._avg_points_idx = (self._avg_points_idx + 1) % self.average_frames

            vertical = sum([x[0] for x in self._avg_points[0:self.average_frames]]) / self.average_frames
            horizontal = sum([x[1] for x in self._avg_points[0:self.average_frames]]) / self.average_frames

            vertical -=  frame.shape[0]/2
            horizontal -=  frame.shape[1]/2
        else:
            vertical = 0
            horizontal = 0

        return ndimage.affine_transform(frame,
                       matrix=[1, 1, 1],
                       offset=[-vertical, -horizontal, 0],
                       order=0)

class Affine:
    def __init__(self, matrix=[[1,0],[0,1]], offset=[0,0], relative=False,
                 *args, **kwargs):

        self.matrix = matrix
        self.offset = offset
        self.relative = relative

        assert(len(matrix) == 2)
        assert(len(matrix[0]) == 2)
        assert(len(matrix[1]) == 2)
        assert(len(offset) == 2)

    def apply(self, *args, **kwargs):
        frame = kwargs['frame']

        matrix = np.zeros((3, 3))
        matrix[0,:2] = self.matrix[0]
        matrix[1,:2] = self.matrix[1]
        matrix[2,2] = 1.0
        offset = self.offset + [0]

        if frame.shape[2] == 3:
            # Add alpha channel
            frame = np.append(frame,
                    np.ones((frame.shape[0], frame.shape[1], 1)) * 255.0,
                    axis=2)

        frame = ndimage.affine_transform(frame,
                       matrix=matrix,
                       offset=offset,
                       order=0)
        return frame


filters.register_filter("flip", Flip)
filters.register_filter("zoom", Zoom)
filters.register_filter("move", Move)
filters.register_filter("translate_to_head", Translate_to_head)
filters.register_filter("affine", Affine)
