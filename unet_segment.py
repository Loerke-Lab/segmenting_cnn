#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:58:35 2021

@author: liamrussell17
"""

from unet_model import build_unet
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

IMAGE_NUM = 150
MODEL_NUM = 2


''' Uncomment below to load model of index MODEL_NUM '''
checkpoint_path = f'/Users/liamrussell17/Desktop/Fall Quarter Lab Rotation/Segmenting Neural Network/saved_model/saved_model_{MODEL_NUM}.ckpt'

input_shape = (512, 512, 1)

model = build_unet(input_shape)
model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights(checkpoint_path)

model.summary()


'''Uncomment the code below to perform a forward pass on image index IMAGE_NUM '''
image_path = f'/Users/liamrussell17/Desktop/Fall Quarter Lab Rotation/Synthetic Images/Gaussian_Filter_Images/image{IMAGE_NUM}.png'

# # real cell image
# # image_path = '/Users/liamrussell17/Desktop/Fall Quarter Lab Rotation/germband_images/test/1/img_000000504_GFP_000.tif'

image = cv2.imread(image_path, 0)
image = np.asarray(image)
image = np.expand_dims(image, 0)
image = image.T.astype(np.uint8)
# image = np.expand_dims(image,0)

print(image.shape)


# prediction = (model.predict(image)[0,:,:,0] > 0.2).astype(np.uint8)

# plt.figure(figsize=(16, 8))
# plt.subplot(221)
# plt.title('Testing Image')
# plt.imshow(image[0,:,:], cmap='gray')
# plt.subplot(222)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')

# plt.show()


''' Uncomment below to summarize filter shapes'''
# layer = model.layers
# filters, biases = model.layers[1].get_weights()
# print(layer[1].name, filters.shape)

# fig1 = plt.figure(figsize=(8,12))
# columns = 6
# rows = 4
# num_filters = 24
# for i in range(1, num_filters + 1):
#     f = filters[:, :, :, i - 1]
#     fig1 = plt.subplot(rows, columns, i)
#     fig1.set_xticks([])
#     fig1.set_yticks([])
#     plt.imshow(f[:, :, 0], cmap='gray')

# plt.show()


'''Uncomment below to summarize filter outputs on image'''
# conv_layer_index = [1,3,6]
# outputs = [model.layers[i].output for i in conv_layer_index]
# model_short = Model(model.inputs, outputs = outputs)

# feature_output = model_short.predict(image)

# columns = 6
# rows = 4
# for ftr in feature_output:
#     fig = plt.figure(figsize=(12,12))
#     for i in range(1, columns*rows + 1):
#         fig = plt.subplot(rows, columns, i)
#         fig.set_xticks([])
#         fig.set_yticks([])
#         plt.imshow(ftr[0,:,:,i-1], cmap='gray')
#     plt.show()

    
    
    