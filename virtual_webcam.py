#!/usr/bin/env python3

import sys
import os
import yaml

import tensorflow as tf
import tfjs_graph_converter.api as tfjs_api
import tfjs_graph_converter.util as tfjs_util

import numpy as np
import cv2
from pyfakewebcam import FakeWebcam

from bodypix_functions import calc_padding
from bodypix_functions import scale_and_crop_to_input_tensor_shape
from bodypix_functions import to_input_resolution_height_and_width
from bodypix_functions import to_mask_tensor

import filters
import time


def load_config(config_mtime, oldconfig={}):
    """
        Load the config file. This only reads the file,
        when its mtime is changed.
    """

    config = oldconfig
    try:
        config_mtime_new = os.stat("config.yaml").st_mtime
        if config_mtime_new != config_mtime:
            print("Reloading config.")
            config = {}
            with open("config.yaml", "r") as configfile:
                yconfig = yaml.load(configfile, Loader=yaml.SafeLoader)
                for key in yconfig:
                    config[key] = yconfig[key]
            config_mtime = config_mtime_new
    except OSError:
        pass
    return config, config_mtime


def reload_layers(config):
    layers = []
    for layer_filters in config.get("layers", []):
        assert(type(layer_filters) == dict)
        assert(len(layer_filters) == 1)
        layer_type = list(layer_filters.keys())[0]
        layer_filters = layer_filters[layer_type]
        layers.append((layer_type, filters.get_filters(config, layer_filters)))
    return layers


# ### Global variables ###

# The last mask frames are kept to average the actual mask
# to reduce flickering
masks = []

# Load the config
config, config_mtime = load_config(0)

# ### End global variables ####


# Set allow_growth for all GPUs
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# tf.get_logger().setLevel("DEBUG")

# VideoCapture for the real webcam
cap = cv2.VideoCapture(config.get("real_video_device"))

if config.get("mjpeg"):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# Configure the resolution and framerate of the real webcam
if config.get("width"):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width"))
if config.get("height"):
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height"))
if config.get("fps"):
    cap.set(cv2.CAP_PROP_FPS, config.get("fps"))

# Attempt to reduce the buffer size
if not cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):
    print('Failed to reduce capture buffer size. Latency will be higher!')

# Get the actual resolution (either webcam default or the configured one)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

config['width'], config['height'] = width, height

# Initialize a fake video device with the same resolution as the real device
fakewebcam = FakeWebcam(config.get("virtual_video_device"), width, height)

# Choose the bodypix (mobilenet) model
# Allowed values:
# - Stride 8 or 16
# internal_resolution: 0.25, 0.5, 0.75, 1.0

output_stride = config.get("stride", 16)
multiplier = config.get("multiplier", 0.5)
model_type = config.get("model", "mobilenet")

if model_type == "resnet":
    model_type = "resnet50"

if model_type == "mobilenet":
    print("Model: mobilenet (multiplier={multiplier}, stride={stride})".format(
        multiplier=multiplier, stride=output_stride))
    model_path = ('bodypix_mobilenet_float_{multiplier:03d}' +
        '_model-stride{stride}').format(
        multiplier=int(100 * multiplier), stride=output_stride)
elif model_type == "resnet50":
    print("Model: resnet50 (stride={stride})".format(
        stride=output_stride))
    model_path = 'bodypix_resnet50_float_model-stride{stride}'.format(
        stride=output_stride)
else:
    print('Unknown model type. Use "mobilenet" or "resnet50".')
    sys.exit(1)

# Load the tensorflow model
print("Loading model...")
graph = tfjs_api.load_graph_model(model_path)
print("done.")

# Setup the tensorflow session
sess = tf.compat.v1.Session(graph=graph)

input_tensor_names = tfjs_util.get_input_tensors(graph)
output_tensor_names = tfjs_util.get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

# Initialize layers
layers = reload_layers(config)

static_image = None
for extension in ["jpg", "jpeg", "png"]:
    if config['real_video_device'].lower().endswith(extension):
        success, static_image = cap.read()

frameNo = 0

def mainloop():
    global config, masks, layers, config_mtime, frameNo
    time_0_start = time.time()

    config, config_mtime_new = load_config(config_mtime, config)
    if config_mtime != config_mtime_new:
        config['width'] = width
        config['height'] = height
        layers = []  # Allow filters to run their destructors
        layers = reload_layers(config)
        config_mtime = config_mtime_new

    time_1_config = time.time()

    if static_image is not None:
        success, frame = True, static_image
    else:
        success, frame = cap.read()
    if not success:
        print("Error getting a webcam image!")
        sys.exit(1)
    # BGR to RGB
    frame = frame[...,::-1]
    frame = frame.astype(np.float)

    time_2_capture = time.time()

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

    time_3_resize = time.time()

    # Preprocessing
    if model_type == "mobilenet":
        resized_frame = np.divide(resized_frame, 127.5)
        resized_frame = np.subtract(resized_frame, 1.0)
    elif model_type == "resnet50":
        m = np.array([-123.15, -115.90, -103.06])
        resized_frame = np.add(resized_frame, m)
    else:
        assert(False)

    sample_image = resized_frame[tf.newaxis, ...]

    time_4_preprocess = time.time()

    results = sess.run(output_tensor_names,
                       feed_dict={input_tensor: sample_image})

    time_5_network = time.time()
    
    if model_type == "mobilenet":
        segment_logits = results[1]
        part_heatmaps  = results[2]
        heatmaps       = results[4]
    else:
        segment_logits = results[6]
        part_heatmaps  = results[5]
        heatmaps       = results[2]

    time_6_result1 = time.time()
        
    scaled_segment_scores = scale_and_crop_to_input_tensor_shape(
        segment_logits, input_height, input_width,
        padT, padB, padL, padR, True
    )
    
    time_6_result2 = time.time()

    # not needed when doing only background replace
    # scaled_part_heatmap_scores = scale_and_crop_to_input_tensor_shape(
    #     part_heatmaps, input_height, input_width,
    #     padT, padB, padL, padR, True
    # )

    time_6_result3 = time.time()
    
    # not needed when doing only background replace
    # scaled_heatmap_scores = scale_and_crop_to_input_tensor_shape(
    #     heatmaps, input_height, input_width,
    #     padT, padB, padL, padR, True
    # )

    time_6_result4 = time.time()

    mask = to_mask_tensor(scaled_segment_scores,
                          config.get("segmentation_threshold", 0.75))
    
    time_6_result5 = time.time()

    mask = np.reshape(mask, mask.shape[:2])

    time_6_result6 = time.time()

    # not needed when doing only background replace
    #part_masks = to_mask_tensor(scaled_part_heatmap_scores, 0.999)
    #part_masks = np.array(part_masks)
    #heatmap_masks = to_mask_tensor(scaled_heatmap_scores, 0.99)
    #heatmap_masks = np.array(heatmap_masks)
    part_masks = None
    heatmap_masks = None

    time_6_result = time.time()

    # Average over the last N masks to reduce flickering
    # (at the cost of seeing afterimages)
    num_average_masks = max(1, config.get("average_masks", 3))
    masks.insert(0, mask)
    masks = masks[:num_average_masks]

    mask = np.mean(masks, axis=0)
    mask = (mask * 255).astype(np.uint8)

    dilate_value = config.get("dilate", 0)
    erode_value = config.get("erode", 0)
    blur_value = config.get("blur", 0)

    if dilate_value:
        mask = cv2.dilate(mask,
                          np.ones((dilate_value, dilate_value), np.uint8),
                          iterations=1)
    if erode_value:
        mask = cv2.erode(mask,
                         np.ones((erode_value, erode_value),
                                 np.uint8), iterations=1)
    if blur_value:
        mask = cv2.blur(mask, (blur_value, blur_value))

    frame = np.append(frame, np.expand_dims(mask, axis=2), axis=2)

    time_7_postprocess = time.time()

    input_frame = frame.copy()
    frame = np.zeros(input_frame.shape)
    for layer_type, layer_filters in layers:
        # Initialize the layer frame
        layer_frame = np.zeros(frame.shape)  # transparent black
        if layer_type == "foreground":
            layer_frame = input_frame.copy()
        elif layer_type == "input":
            layer_frame = input_frame.copy()
            # make the frame opaque
            layer_frame[:,:,3] = 255 * np.ones(input_frame.shape[:2])
        elif layer_type == "previous":
            layer_frame = frame.copy()
            # make the frame opaque
            layer_frame[:,:,3] = 255 * np.ones(input_frame.shape[:2])
        elif layer_type == "empty":
            pass

        layer_frame = filters.apply_filters(layer_frame, mask, part_masks,
                                            heatmap_masks, layer_filters)
        if layer_frame.shape[2] == 4:
            transparency = layer_frame[:,:,3] / 255.0
            transparency = np.expand_dims(transparency, axis=2)
            frame[:,:,:3] = frame[:,:,:3] * \
                (1.0 - transparency) + layer_frame[:,:,:3] * transparency
        else:
            frame[:,:,:3] = layer_frame[:,:,:3].copy()

    time_8_compose = time.time()

    # Remove alpha channel
    frame = frame[:,:,:3]

    if config.get("debug_show_mask") is not None:
        mask_id = int(config.get("debug_show_mask", None))
        if mask_id >-1 and mask_id < 24:
            mask = part_masks[:,:,mask_id] * 255.0
        frame[:,:,0] = mask
        frame[:,:,1] = mask
        frame[:,:,2] = mask
    elif config.get("debug_show_heatmap") is not None:
        heatmap_id = int(config.get("debug_show_heatmap", None))
        if heatmap_id >-1 and heatmap_id < 17:
            mask = heatmap_masks[:,:,heatmap_id] * 255.0
        frame[:,:,0] = mask
        frame[:,:,1] = mask
        frame[:,:,2] = mask

    frame = frame.astype(np.uint8)
    fakewebcam.schedule_frame(frame)

    time_9_show = time.time()
    # Output the theoretical FPS for each processing stage
    if (frameNo % 100) == 0:
        print("Config {}".format(1/(time_1_config - time_0_start)))
        print("Capture {}".format(1/(time_2_capture - time_1_config)))
        print("Resize {}".format(1/(time_3_resize - time_2_capture)))
        print("Preprocess {}".format(1/(time_4_preprocess - time_3_resize)))
        print("Network {}".format(1/(time_5_network - time_4_preprocess)))
        print("Result {}".format(1/(time_6_result - time_5_network)))
        print("Result 1 {}".format(1/(time_6_result1 - time_5_network)))
        print("Result 2 {}".format(1/(time_6_result2 - time_6_result1)))
        #print("Result 3 {}".format(1/(time_6_result3 - time_6_result2)))
        #print("Result 4 {}".format(1/(time_6_result4 - time_6_result3)))
        print("Result 5 {}".format(1/(time_6_result5 - time_6_result4)))
        print("Result 6 {}".format(1/(time_6_result6 - time_6_result5)))
        print("Postprocess {}".format(1/(time_7_postprocess - time_6_result)))
        print("Compose {}".format(1/(time_8_compose - time_7_postprocess)))
        print("Show {}".format(1/(time_9_show - time_8_compose)))
        print("Total {}".format(1/(time_9_show - time_0_start)))
        print()
    frameNo = frameNo + 1

if __name__ == "__main__":
    while True:
        try:
            mainloop()
        except KeyboardInterrupt:
            print("stopping.")
            break
