#!/bin/sh
mkdir bodypix_mobilenet_float_050_model-stride16
wget -O bodypix_mobilenet_float_050_model-stride16/model.json https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/050/model-stride16.json
wget -O bodypix_mobilenet_float_050_model-stride16/group1-shard1of1.bin https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/050/group1-shard1of1.bin
