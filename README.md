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
except for `width` and `height` as the webcam must be reinitialized to change them and
`multiplier` and `output_stride` as the model must be reloaded to apply them.

- `width`: The input resolution width.
- `height`: The input resolution height.
- `segmentation_threshold`: The threshold for foreground / background segmentation.
- `blur`: Blur factor for the mask to smooth the edges.
- `dilate`: Number of pixels the mask is shrunk to remove spots.
- `erode`: Number of pixels the mask is grown after shrinking to capture the full body image again.
- `real_video_device`: The video device of your webcam, e.g. `/dev/video0`.
- `average_masks`: Number of masks to average. A higher number will result in afterimages,
  a smaller number in flickering at the boundary between foreground and background.
- `layers`: A list of videos layers like the input webcam image, the segmented foreground,
  virtual backgrounds or image overlays.
- `debug_show_mask`: Debug option to show the mask, that can be used to configure
  blur/dilate/erode correctly.
- `multiplier`: Multiplier parameter of the model (0.5, 0.75 or 1.0). You need to download the
  matching model when you change this parameter.
- `output_stride`: Stride parameter of the model (16 or 8). You need to download the matching model
  when you change the parameter.
- `internal_resolution`: Resolution factor (between 0.0 and 1.0) for the model input. Smaller is
  faster and less accurate. Note that 1.0 does not always give the best results.

Note: Input `width` and `height` are autodetected when they are not set in the config,
but this can lead to bad default values, e.g., `640x480` even when the camera supports
a resolution of `1280x720`.

## Layers

The layers option contains one image source and a list of filters. The image sources are:

- `input`: The webcam image
- `foreground`: The foreground of the image, i.e., the person.
- `previous`: The image composed of all previous layers.
- `empty`: A transparent image.

Each layer has a list of filters, that are applied in the given order.
After all filters are applied, the layer is merged with the previous layers.

## Filters

Each layer has a list of filters.
A simple example that converts the background to grayscale and blurs it looks like this:

	  - input: ["grayscale", "blur"]

Some filters have arguments. To change the blur value in the filter list above,
you can use onf of these syntax variants:

- Flat list: `["grayscale", ["blur", 10, 10]]`
- Argument list: `["grayscale", ["blur", [10, 10]]]`
- Keyword arguments: `["grayscale", ["blur", {intensity_x: 10, intensity_y: 10}]]`


## Example Layers

#### A virtual background

	- layers:
	  - empty: [["image", "background.jpg"]]
	  - foreground: []

#### Blurred background

	- layers:
	  - input: [["blur", 10]]
	  - foreground: []

#### Blurred background and a moving fog overlay

	- layers:
	  - input: [["blur", 10]]
	  - foreground: []
	  - previous: [["image", "images/fog.jpg"], ["roll", 5, 0]]

### Filters

The current filters and their options are:

- `image`: Returns a static image, e.g., to use a virtual background.
  - `image_path`: The path to the image file.
  - `interpolation_method`: The interpolation method. Currently are `LINEAR` and `NEAREST` supported and `LINEAR` is the default.
    When you use a pixel art background, it may look better with `NEAREST`.
- `image_sequence`: Returns images from an image sequence. This can be used for animated backgrounds or overlays.
  - `images_path`: The path to a folder containing the images. The folder must only contain images and they must have the correct order
    when they are sorted by filename.
  - `fps`: The frames per second of the animation.
  - `interpolation_method`: `LINEAR` or `NEAREST` interpolation
- `video`: Returns images from a video. This can be used for animated backgrounds or overlays.
  - `video_path`: The path to the video.
  - `target_fps`: The target frames per second of image sequence generated from the video.
    This can be used to reduce the RAM usage.
  - `interpolation_method`: `LINEAR` or `NEAREST` interpolation
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
- `stripes`: Add a semi-transparent stripe effect with darker and lighter stripes.
  - `width`: Width of a stripe.
  - `intensity`: Intensity how much darker/lighter the stripe is.
  - `speed`: Speed at which the stripes move across the image.
- `chroma_key`: Convert a color to transparency (green screen effect).
  - `r`: Red channel of the color.
  - `g`: Green channel of the color.
  - `b`: Blue channel of the color.
  - `fuzz`: Factor for fuzzy matching of similar colors.

## Animations

Animations are a sequence of images in a folder. The script simply trys to load `image_name/*.*`
when `image_name` is a folder. You need to make sure, that the folder only contains images and that
the images are ordered correctly when they are ordered by their filename.

Example for creating an animation from a short video:

    mkdir animation
    cd animation
    ffmpeg -i ../animation.gif -vf fps=10 out%04d.png

When using the `ffmpeg` command, you can change the framerate of the video using the `fps` parameter.

Note that the script loads all images of an animation into RAM scaled to the resolution of your webcam, so
using too long animations is not a good idea.

## Advanced

You can try alternative neural network models by editing the source code.

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
