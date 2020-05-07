import cv2
import filters
import os
import stat
import glob
import time


def reload_images(images_path, mtime, width, height, interpolation_method):
    # Do nothing, if the image is unchanged
    images_stat = os.stat(images_path)
    if images_stat.st_mtime == mtime:
        return None, mtime
    mtime_new = images_stat.st_mtime

    if stat.S_ISDIR(images_stat.st_mode):
        filenames = glob.glob(images_path + "/*.*")
    else:
        filenames = [images_path]

    images = []
    for filename in filenames:
        image_raw = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        _interpolation_method = cv2.INTER_LINEAR
        if interpolation_method == "NEAREST":
            _interpolation_method = cv2.INTER_NEAREST

        image = cv2.resize(image_raw, (width, height),
                           interpolation=_interpolation_method)
        if len(image.shape) == 2:  # grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR to RGB
        image[:,:,0], image[:,:,2] = image[:,:,2], image[:,:,0].copy()
        images.append(image)

    return images, mtime_new


class Image:
    def __init__(self, image_path, interpolation_method="LINEAR",
                 *args, **kwargs):
        config = kwargs['config']
        self.width = config.get("width")
        self.height = config.get("height")
        assert(self.width is not None and self.height is not None)

        self.image_path = image_path
        self.interpolation_method = interpolation_method
        self.mtime = 0

        self.reload_image()

    def reload_image(self):
        images, new_mtime = reload_images(self.image_path, self.mtime,
                self.width, self.height, self.interpolation_method)

        if images:
            self.image = images[0]
            self.mtime = new_mtime

    def apply(self, *args, **kwargs):
        self.reload_image()
        return self.image.copy()


class ImageSequence:
    def __init__(self, images_path, fps=10, interpolation_method="LINEAR",
                 *args, **kwargs):
        config = kwargs['config']
        self.width = config.get("width")
        self.height = config.get("height")
        assert(self.width is not None and self.height is not None)

        self.images_path = images_path
        self.fps = fps
        self.interpolation_method = interpolation_method
        self.mtime = 0

        self.reload_images()

    def reload_images(self):
        images, new_mtime = reload_images(self.images_path, self.mtime,
                self.width, self.height, self.interpolation_method)

        if images:
            self.images = images
            self.mtime = new_mtime
            self.idx = 0
            self.last_frame_time = time.time()

    def apply(self, *args, **kwargs):
        self.reload_images()
        frame = self.images[self.idx].copy()
        if time.time() - self.last_frame_time > 1.0 / self.fps:
            self.idx = (self.idx + 1) % len(self.images)
            self.last_frame_time = time.time()
        return frame


filters.register_filter("image", Image)
filters.register_filter("image_sequence", ImageSequence)
