import os
import yaml
import stat
import time
import cv2
import glob

def load_config(data, oldconfig={}):
    """
        Load the config file. This only reads the file,
        when its mtime is changed.
    """

    config = oldconfig
    try:
        if os.stat("config.yaml").st_mtime != data.get("config_mtime"):
            print("Reloading config.")
            config = {}
            with open("config.yaml", "r") as configfile:
                yconfig = yaml.load(configfile, Loader=yaml.SafeLoader)
                for key in yconfig:
                    config[key] = yconfig[key]
            # Force image reload
            for key in data:
                if key.endswith("_mtime"):
                    data[key] = 0
            data["config_mtime"] = os.stat("config.yaml").st_mtime
    except OSError:
        pass
    return config


def load_images(images, image_name, height, width, imageset_name, data,
                interpolation_method="NEAREST", image_filters=[]):
    """
        Load and preprocess image(s)
        image_name must be either the path to an image file or
        the path to an folder containing multiple files that should be
        played as animation.
        imageset_name is an unique name that is used to store values like
        the mtime in data
        The function only reloads the image(s) when the mtime of the file
        or folder is changed.
    """
    try:
        replacement_stat = os.stat(image_name)
        if replacement_stat.st_mtime != data.get(imageset_name + "_mtime"):
            time.sleep(0.1)
            print("Loading images {0} ...".format(image_name))
            data[imageset_name + "_idx"] = 0
            filenames = [image_name]
            if stat.S_ISDIR(replacement_stat.st_mode):
                filenames = glob.glob(filenames[0] + "/*.*")
                if not filenames:
                    return None

            images = []
            for filename in filenames:
                image_raw = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                interpolation_method = cv2.INTER_LINEAR
                if interpolation_method == "NEAREST":
                    interpolation_method = cv2.INTER_NEAREST
                image = cv2.resize(image_raw, (width, height),
                    interpolation=interpolation_method)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # BGR to RGB
                image[:,:,0], image[:,:,2] = image[:,:,2], image[:,:,0].copy()
                images.append(image)

            data[imageset_name + "_mtime"] = os.stat(image_name).st_mtime

            for i in range(len(images)):
                for image_filter in image_filters:
                    images[i] = image_filter(frame=images[i])
            print("Finished loading images.")

        return images

    except OSError:
        return None


