#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:33:28 2021

@author: liamrussell17
"""

# Building Unet by dividing encoder and decoder into blocks

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate

FILTER_SIZE = 5

def conv_block(input, num_filters):
    x = Conv2D(num_filters, FILTER_SIZE, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, FILTER_SIZE, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x


#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   


#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 24)
    s2, p2 = encoder_block(p1, 48)
    s3, p3 = encoder_block(p2, 96)
    s4, p4 = encoder_block(p3, 192)

    b1 = conv_block(p4, 384) #Bridge

    d1 = decoder_block(b1, s4, 192)
    d2 = decoder_block(d1, s3, 96)
    d3 = decoder_block(d2, s2, 48)
    d4 = decoder_block(d3, s1, 24)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model





