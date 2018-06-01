#!/bin/sh

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -ex

# ResNet 50,101,152

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# ResNet 50 v1 (self-trained)

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-50-v1-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-50-v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

# Inception-BN

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# VGG 16,19

python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-symbol.json --param-file=./model/imagenet1k-vgg-16-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-vgg-16-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-vgg-16-quantized-0000.params --label-name='prob_label' --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Inception-V3,V4 (self-trained)

python imagenet_inference.py --symbol-file=./model/Inception-7-symbol.json --param-file=./model/Inception-7-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-5batches-naive-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-10batches-naive-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-50batches-naive-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-5batches-entropy-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-10batches-entropy-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/Inception-7-quantized-50batches-entropy-symbol.json --param-file=./model/Inception-7-quantized-0000.params --image-shape='3,299,299' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/mydata_299_299_val.rec --ctx=cpu

# Squeezenet 1.0,1.1

python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu
python imagenet_inference.py --symbol-file=./model/imagenet1k-squeezenet-v1_0-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-squeezenet-v1_0-quantized-0000.params --label-name='prob_label' --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# MobileNet
python imagenet_inference.py --symbol-file=./model/imagenet1k-mobilenet-v1-symbol.json --param-file=./model/imagenet1k-mobilenet-v1-0000.params --rgb-mean=123.68,116.779,103.939 --resize=256 --scale=0.017 --num-skipped-batches=50 --num-inference-batches=1 --dataset=./data/val_256_q90.rec --ctx=cpu
