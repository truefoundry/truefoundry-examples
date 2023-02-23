#!/bin/bash

wget -O mobilenet_v3_small_075_224.tar.gz https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5?tf-hub-format=compressed
tar -C models/imagenet_mobilenet_v3_small_075_224_classification_5/1/ -zxvf mobilenet_v3_small_075_224.tar.gz
rm mobilenet_v3_small_075_224.tar.gz
