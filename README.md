# Virtual Webcam Background

Virtual Webcam Background allows you to use a virtual background or to blur the background of your webcam image similar to commercial programs like Zoom.

Tensorflow with [BodyPix](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html) is used to segment the image into foreground (person) and background using a neural network and [v4l2loopback](https://github.com/umlaeute/v4l2loopback) is used to create a virtual webcam.

As the script creates a virtual webcam device it works with any program that can use a v4l2 webcam.

See [virtual-webcam.com](https://www.virtual-webcam.com) for more information, image packs and more.

## Installation

The program needs python3 and is tested with python 3.7.

Install the requirements:

    pip install -r requirements.txt

Download the bodypix model:

    ./get-model.sh

Then install v4l2loopback and load the kernel module:

    modprobe v4l2loopback exclusive_caps=1 video_nr=2 # creates /dev/video2

The `exclusive_caps` option is needed by some programs, such as chromium.

Then copy `config.yaml.example` to `config.yaml` and edit the config as needed and run
the `virtual_webcam.py` script.

If you have a Nvidia graphics card, you may want to install CUDA for better performance.

## Configuration

To configure the virtual webcam, edit `config.yaml`. Most options are applied instantly,
except for `width` and `height` as the webcam must be reinitialized to change them.

- `width`: The input resolution width.
- `height`: The input resolution height.
- `segmentation_threshold`: The threshold for foreground / background segmentation.
- `blur`: Blur factor for the mask to smooth the edges.
- `dilate`: Number of pixels the mask is shrunk to remove spots.
- `erode`: Number of pixels the mask is grown after shrinking to capture the full body image again.
- `background_filters`: Filters applied to the background or virtual background.
- `foreground_filters`: Filters applied to the detected foreground.
- `result_filters`: Filters applied to the result image.
- `background_image`: Filename of an image file or a directory containing images for an animation.
- `overlay_image`: Filename of an image file with alpha channel (transparency) or a directory
  containing images for an animation.
- `background_fps`: Maximal framerate for the animation of the background image.
- `overlay_fps`: Maximal framerate for the animation of the overlay image.
- `virtual_video_device`: The virtual video device, e.g., `/dev/video2`.
- `real_video_device`: The video device of your webcam, e.g. `/dev/video0`.
- `average_masks`: Number of masks to average. A higher number will result in afterimages,
  a smaller number in flickering at the boundary between foreground and background.
- `background_interpolation_method`: Interpolation method to use. Currently supported methods
  are `BILINEAR` and `NEAREST`. Usually `BILINEAR` is an good option, but for using pixel art
  backgrounds `NEAREST` may look better.
- `debug_show_mask`: Debug option to show the mask, that can be used to configure
  blur/dilate/erode correctly.

Note: Input `width` and `height` are autodetected when they are not set in the config,
but this can lead to bad default values, e.g. `640x480` even when the camera supports
a resolution of `1280x720`.

## Filters

The `background_filters` option is a list of filters that will be applied after each other.

A simple example that converts the background to grayscale and blurs it:

```
- background_filters = ["grayscale", "blur"]
```

Some filters have arguments. To change the blur value in the filter list above, use

```
- background_filters = ["grayscale", ["blur", 10, 10]]
```

Alternative syntax variants:

```
- background_filters = ["grayscale", ["blur", [10, 10]]]
- background_filters = ["grayscale", ["blur", {intensity_x: 10, intensity_y: 10}]]
```

### Filters

The current filters and their options are:

- `blur`: Blur the image.i
  - `intensity_x`: The intensity in the x direction.
  - `intensity_y`: The intensity in the y direction. When only `intensity_x` is given, it will be used for `intensity_y` as well.
- `gaussian_blur`: Blur the image using a Gaussian blur. It looks better than normal box blur, but is more CPU intensive.
  - `intensity_x`: The intensity in the x direction. Must be an odd value: even values are bumped to the next odd value.
  - `intensity_y`: The intensity in the y direction. Must be an odd value: even values are bumped to the next odd value. When only `intensity_x` is given, it will be used for `intensity_y` as well.
- `grayscale`: Convert the image into a grayscale image.
- `roll`: move an image with a constant speed. This is mostly useful for overlays.
  - `speed_x`: Speed in x direction.
  - `speed_y`: Speed in y direction.
- `change_alpha`: Change the transparency of an image.
  - `change_alpha`: Alpha value to add (between `-255` and `255`)
  - `alpha_min`, `alpha_max`: Transparency levels to clip the resulting alpha value.
- `single_color`: Change the image to grayscale and then color it with a given color.
  - `r`, `g`, `b`: RGB values.
- `color_filter`: Change the color levels by multiplying the RGB values with a factor between `0` and `255`.
  - `r`, `g`, `b`: The factors for the colors red/green/blue.
- `flip`: Flip the image horizontally or vertically.
  - `horizontal`: Flip horizontally.
  - `vertical`: Flip vertically.

## Animations

Animations are a sequence of images in a folder. The script simply trys to load `image_name/*.*`
when `image_name` is a folder. You need to make sure, that the folder only contains images and that
the images are ordered correctly when they are ordered by their filename.

Example for creating an animation from a short video:

    mkdir animation
    cd animation
    ffmpeg -i ../animation.gif -vf fps=10 out%04d.png

Animations currently run at the framerate of the virtual webcam, so their speed depends on the framerate
that your computer is able to achieve for the virtual webcam. When using the `ffmpeg` command, you can
adapt the framerate of the video to the framerate of the virtual webcam using the `fps` parameter.

Note that the script loads all images of an animation into RAM scaled to the resolution of your webcam, so
using too long animations is not a good idea.

## Advanced

You can try alternative models by editing the source code.

To download other models get the full `get-model.sh` script from [https://github.com/ajaichemmanam/simple\_bodypix\_python](https://github.com/ajaichemmanam/simple_bodypix_python) and run it with one of these combinations:

    ./get-model.sh bodypix/mobilenet/float/{025,050,075,100}/model-stride{8,16}

Then edit the script and change `output_stride` and `internal_resolution` accordingly.

You can also try the `resnet50` models, but then you will in addition need to change the preprocessing.
The needed preprocessing for resnet50 is included as a comment in the source code.

## Acknowledgements

- The program is inspired by this [blog post](https://elder.dev/posts/open-source-virtual-background/) by Benjamin Elder.
- [Linux-Fake-Background-Webcam](https://github.com/fangfufu/Linux-Fake-Background-Webcam) is a direct implementation of the blog post using docker and nodejs in addition to python.
- [simple\_bodypix\_python](https://github.com/ajaichemmanam/simple_bodypix_python) was the base for an earlier version of the script.
- The functions in `bodypix_functions.py` are adapted from the [body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix) nodejs module.
