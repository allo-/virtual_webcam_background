import tensorflow as tf
import numpy as np

def remove_padding_and_resize_back(resized_and_padded, original_height, original_width,
        padT, padB, padL, padR):
    return tf.squeeze(tf.image.crop_and_resize(resized_and_padded,
        [[padT / (original_height + padT + padB - 1.0),
        padL / (original_width + padL + padR - 1.0),
        (padT + original_height - 1.0) / (original_height + padT + padB - 1.0),
        (padL + original_width - 1.0) / (original_width + padL + padR - 1.0)]],
        [0], [original_height, original_width]
    ), [0])

def scale_and_crop_to_input_tensor_shape(tensor,
        input_tensor_height, input_tensor_width,
        padT, padB, padL, padR,
        apply_sigmoid_activation):
    in_resized_and_padded = tf.image.resize_with_pad(tensor,
        input_tensor_height, input_tensor_width,
        method=tf.image.ResizeMethod.BILINEAR)
    if apply_sigmoid_activation:
        in_resized_and_padded = tf.sigmoid(in_resized_and_padded)

    return remove_padding_and_resize_back(in_resized_and_padded,
        input_tensor_height, input_tensor_width, padT, padB, padL, padR)

def is_valid_input_resolution(resolution, output_stride):
    return (resolution - 1) % output_stride == 0;

def to_valid_input_resolution(input_resolution, output_stride):
    if is_valid_input_resolution(input_resolution, output_stride):
        return input_resolution
    return int(np.floor(input_resolution / output_stride) * output_stride + 1)

def to_input_resolution_height_and_width(internal_resolution, output_stride, input_height, input_width):
    return (to_valid_input_resolution(input_height * internal_resolution, output_stride),
            to_valid_input_resolution(input_width * internal_resolution, output_stride))

def to_mask_tensor(segment_scores, threshold):
    return tf.math.greater(segment_scores, tf.constant(threshold))

def calc_padding(input_tensor, targetH, targetW):
    height, width = input_tensor.shape[:2]
    target_aspect = targetW / targetH;
    aspect = width / height;
    padT, padB, padL, padR = 0, 0, 0, 0;
    if aspect < target_aspect:
        padT = 0
        padB = 0
        padL = round(0.5 * (target_aspect * height - width))
        padR = round(0.5 * (target_aspect * height - width))
    else:
        padT = round(0.5 * ((1.0 / target_aspect) * width - height))
        padB = round(0.5 * ((1.0 / target_aspect) * width - height))
        padL = 0
        padR = 0
    return padT, padB, padL, padR
