#!/usr/bin/env python3

import tensorflow as tf
import cv2
import sys
import tfjs_graph_converter.api as tfjs_api
import tfjs_graph_converter.util as tfjs_util
import numpy as np
import time
from pyfakewebcam import FakeWebcam

from bodypix_functions import calc_padding
from bodypix_functions import scale_and_crop_to_input_tensor_shape
from bodypix_functions import to_input_resolution_height_and_width
from bodypix_functions import to_mask_tensor

import filters
from loader import load_config, load_images

# Default config values
config = {}

data = {}

### Global variables ###

# Background frames and the current index in the list
# when the background is a animation
replacement_bgs = None

# Overlays
overlays = None
background_overlays = None

# The last mask frames are kept to average the actual mask
# to reduce flickering
masks = []

# Load the config
config, _ = load_config(data)

def reload_filters(config):
    filter_instances = {}
    for name in ["background", "background_overlay", "foreground",
            "overlay", "result"]:
        filter_instances[name] = filters.get_filters(
                config.get(name + "_filters", []))
    return filter_instances

filter_instances = reload_filters(config)

### End global variables ####


# Set allow_growth for all GPUs
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# tf.get_logger().setLevel("DEBUG")

# VideoCapture for the real webcam
cap = cv2.VideoCapture(config.get("real_video_device"))

# Configure the resolution of the real webcam
if config.get("width"):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width"))
if config.get("height"):
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height"))
# cap.set(cv2.CAP_PROP_FPS, 30)

# Get the actual resolution (either webcam default or the configured one)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a fake video device with the same resolution as the real device
fakewebcam = FakeWebcam(config.get("virtual_video_device"), width, height)

# Choose the bodypix (mobilenet) model
# Allowed values:
# - Stride 8 or 16
# internal_resolution: 0.25, 0.5, 0.75, 1.0

output_stride = config.get("stride", 16)
multiplier = config.get("multiplier", 0.5)

model_path = 'bodypix_mobilenet_float_{0:03d}_model-stride{1}'.format(
    int(100 * multiplier), output_stride)

# Load the tensorflow model
print("Loading model...")
graph = tfjs_api.load_graph_model(model_path)
print("done.")

# Setup the tensorflow session
sess = tf.compat.v1.Session(graph=graph)

input_tensor_names = tfjs_util.get_input_tensors(graph)
output_tensor_names = tfjs_util.get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])


def mainloop():
    global config, masks, replacement_bgs, overlays, background_overlays
    global filter_instances

    config, config_changed = load_config(data, config)
    if config_changed:
        filter_instances = reload_filters(config)

    success, frame = cap.read()
    if not success:
        print("Error getting a webcam image!")
        sys.exit(1)
    # BGR to RGB
    frame = frame[...,::-1]
    frame = frame.astype(np.float)

    image_name = config.get("background_image", "background.jpg")
    replacement_bgs, _ = load_images(replacement_bgs, image_name,
        height, width, "replacement_bgs", data,
        config.get("background_interpolation_method"))

    if not replacement_bgs:
        replacement_bgs = [np.copy(frame)]

    input_height, input_width = frame.shape[:2]
    internal_resolution = config.get("internal_resolution", 0.5)

    target_height, target_width = to_input_resolution_height_and_width(
        internal_resolution, output_stride, input_height, input_width)

    padT, padB, padL, padR = calc_padding(frame, target_height, target_width)
    resized_frame = tf.image.resize_with_pad(
        frame,
        target_height, target_width,
        method=tf.image.ResizeMethod.BILINEAR
    )

    resized_height, resized_width = resized_frame.shape[:2]

    # Preprocessing for resnet
    #m = np.array([-123.15, -115.90, -103.06])
    #resized_frame = np.add(resized_frame, m)

    # Preprocessing for mobilenet
    resized_frame = np.divide(resized_frame, 127.5)
    resized_frame = np.subtract(resized_frame, 1.0)
    sample_image = resized_frame[tf.newaxis, ...]

    results = sess.run(output_tensor_names,
        feed_dict={input_tensor: sample_image})
    segments = np.squeeze(results[1], 0)

    segment_logits = results[1]
    scaled_segment_scores = scale_and_crop_to_input_tensor_shape(
        segment_logits, input_height, input_width,
        padT, padB, padL, padR, True
    )

    mask = to_mask_tensor(scaled_segment_scores,
        config.get("segmentation_threshold", 0.75))
    mask = tf.dtypes.cast(mask, tf.int32)
    mask = np.reshape(mask, mask.shape[:2])

    # Average over the last N masks to reduce flickering
    # (at the cost of seeing afterimages)
    masks.insert(0, mask)
    num_average_masks = max(1, config.get("average_masks", 3))
    masks = masks[:num_average_masks]
    mask = np.mean(masks, axis=0)
    mask = (mask * 255).astype(np.uint8)

    dilate_value = config.get("dilate", 0)
    erode_value = config.get("erode", 0)
    blur_value = config.get("blur", 0)

    if dilate_value:
        mask = cv2.dilate(mask,
            np.ones((dilate_value, dilate_value), np.uint8), iterations=1)
    if erode_value:
        mask = cv2.erode(mask,
            np.ones((erode_value, erode_value),
            np.uint8), iterations=1)
    if blur_value:
        mask = cv2.blur(mask, (blur_value, blur_value))

    # Foreground (with the mask in the alpha channel)
    foreground = np.append(frame, np.expand_dims(mask, axis=2), axis=2)
    foreground = filters.apply_filters(foreground,
            filter_instances['foreground'])

    # Background (without mask)
    replacement_bgs_idx = data.get("replacement_bgs_idx", 0)
    background = np.copy(replacement_bgs[replacement_bgs_idx])

    background = filters.apply_filters(background,
            filter_instances['background'])

    background_overlays_idx = data.get("background_overlays_idx", 0)
    background_overlays, _ = load_images(background_overlays,
        config.get("background_overlay_image", ""),
        height, width, "overlays", data)

    # Background overlays
    if background_overlays:
        background_overlay = np.copy(
                background_overlays[background_overlays_idx])

        # Filter the overlay
        background_overlay = filters.apply_filters(background_overlay,
            filter_instances['background_overlay'])

        # The image has an alpha channel
        assert(background_overlay.shape[2] == 4)

        for c in range(3):
            background[:,:,c] = background[:,:,c] * \
                (1.0 - background_overlay[:,:,3] / 255.0) + \
                background_overlay[:,:,c] * (background_overlay[:,:,3] / 255.0)

        time_since_last_frame = time.time() - \
            data.get("last_frame_background_overlay", 0)
        if time_since_last_frame > 1.0 / \
                config.get("background_overlay_fps", 1):
            data["background_overlays_idx"] = \
                (background_overlays_idx + 1) % len(background_overlays)
            data["last_frame_background_overlay"] = time.time()

    # Merge background and foreground (both with mask)
    mask = foreground[:,:,3].astype(np.float) # get the mask from the alpha channel
    mask /= 255.
    mask = np.expand_dims(mask, axis=2)
    mask_inv = 1.0 - mask
    frame = foreground[:,:,:3] * mask + background[:,:,:3] * mask_inv

    time_since_last_frame = time.time() - data.get("last_frame_bg", 0)
    if time_since_last_frame > 1.0 / config.get("background_fps", 1):
        data["replacement_bgs_idx"] = \
            (replacement_bgs_idx + 1) % len(replacement_bgs)
        data["last_frame_bg"] = time.time()

    # Filter the result
    frame = filters.apply_filters(frame,
            filter_instances['result'])

    # Overlays
    overlays_idx = data.get("overlays_idx", 0)
    overlays, _ = load_images(overlays, config.get("overlay_image", ""),
        height, width, "overlays", data)

    if overlays:
        overlay = np.copy(overlays[overlays_idx])

        # Filter the overlay
        overlay = filters.apply_filters(overlay,
            filter_instances['overlay'])

        assert(overlay.shape[2] == 4) # The image has an alpha channel
        for c in range(3):
            frame[:,:,c] = frame[:,:,c] * (1.0 - overlay[:,:,3] / 255.0) + \
                overlay[:,:,c] * (overlay[:,:,3] / 255.0)

        time_since_last_frame = time.time() - data.get("last_frame_overlay", 0)
        if time_since_last_frame > 1.0 / config.get("overlay_fps", 1):
            data["overlays_idx"] = (overlays_idx + 1) % len(overlays)
            data["last_frame_overlay"] = time.time()

    if config.get("debug_show_mask", False):
        frame[:,:] = mask * 255

    frame = frame.astype(np.uint8)
    fakewebcam.schedule_frame(frame)

while True:
    try:
        mainloop()
    except KeyboardInterrupt:
        print("stopping.")
        break
