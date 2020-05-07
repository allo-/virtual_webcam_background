import cv2
import filters
import os
import stat


class Image:
    def __init__(self, image_path, interpolation_method="NEAREST",
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
        # Do nothing, if the image is unchanged
        image_stat = os.stat(self.image_path)
        if image_stat.st_mtime == self.mtime:
            return
        self.mtime = image_stat.st_mtime

        image_raw = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)

        interpolation_method = cv2.INTER_LINEAR
        if self.interpolation_method == "NEAREST":
            interpolation_method = cv2.INTER_NEAREST

        image = cv2.resize(image_raw, (self.width, self.height),
                           interpolation=interpolation_method)
        if len(image.shape) == 2:  # grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR to RGB
        image[:,:,0], image[:,:,2] = image[:,:,2], image[:,:,0].copy()
        self.image = image

    def apply(self, *args, **kwargs):
        self.reload_image()
        return self.image


filters.register_filter("image", Image)
