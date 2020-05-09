import cv2
import filters
import os
import stat
import glob
import time
import numpy as np


def reload_video(video_path, mtime, width, height,
        target_fps, interpolation_method):
    # Do nothing, if the video is unchanged
    video_stat = os.stat(video_path)
    if video_stat.st_mtime == mtime:
        return None, mtime
    mtime_new = video_stat.st_mtime

    print("Loading video: " + video_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if target_fps <= 0 or target_fps > fps:
        target_fps = fps
    every_nth_frame = fps / target_fps

    _interpolation_method = cv2.INTER_LINEAR
    if interpolation_method == "NEAREST":
        _interpolation_method = cv2.INTER_NEAREST

    images = []
    frame_no = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_no += 1
        if (frame_no % every_nth_frame) != 0:
            continue

        image = cv2.resize(frame, (width, height),
                           interpolation=_interpolation_method)

        # BGR to RGB
        image[:,:,0], image[:,:,2] = image[:,:,2], image[:,:,0].copy()
        images.append(image)

    if images:
        print("Finished loading video.")
    else:
        print("Error loading video (format not supported by OpenCV?).")
    return images, mtime_new


class Video:
    def __init__(self, video_path, target_fps=10, interpolation_method="LINEAR",
                 *args, **kwargs):
        config = kwargs['config']
        self.width = config.get("width")
        self.height = config.get("height")
        assert(self.width is not None and self.height is not None)

        self.video_path = video_path
        self.fps = target_fps
        self.interpolation_method = interpolation_method
        self.mtime = 0

        self.reload_video()

    def reload_video(self):
        images, new_mtime = reload_video(self.video_path, self.mtime,
                self.width, self.height, self.fps,
                self.interpolation_method)

        self.images = images
        self.mtime = new_mtime
        if images:
            self.images = images
            self.idx = 0
            self.last_frame_time = time.time()

    def apply(self, *args, **kwargs):
        self.reload_video()

        if not self.images:
            return np.zeros((self.height, self.width, 3))

        frame = self.images[self.idx].copy()
        if time.time() - self.last_frame_time > 1.0 / self.fps:
            self.idx = (self.idx + 1) % len(self.images)
            self.last_frame_time = time.time()
        return frame


filters.register_filter("video", Video)
