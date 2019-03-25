#!/usr/bin/env python
# -*- coding:utf-8

import keras

bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1


def dense_block(x, blocks, growth_rate, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '/block' + str(i + 1))
    return x


def transition_block_down(x, filters, name):
    """A downsampling transition block.
    # Arguments
        x: input tensor.
        filters: int, number of filters in convolutional layer.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.regularizers.l2(1e-4),
                                        name=name + '/bn')(x)
    x = keras.layers.Activation('relu', name=name + '/relu')(x)
    x = keras.layers.Conv2D(filters, 1, use_bias=False, name=name + '/conv')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '/pool')(x)
    return x


def transition_block_up(x, filters, name):
    """An upsampling transition block.
    # Arguments
        x: input tensor.
        filters: int, number of filters in convolutional layer.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.regularizers.l2(1e-4),
                                        name=name + '/bn')(x)
    x = keras.layers.Activation('relu', name=name + '/relu')(x)
    x = keras.layers.Conv2D(filters, 1, use_bias=False, name=name + '/conv')(x)
    x = keras.layers.UpSampling2D(2, interpolation='bilinear', name=name + '/up')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    x1 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                         gamma_regularizer=keras.regularizers.l2(1e-4),
                                         name=name + '_0_bn')(x)
    x1 = keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = keras.layers.Conv2D(4 * growth_rate, 1,
                             use_bias=False,
                             name=name + '_1_conv')(x1)
    x1 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                         gamma_regularizer=keras.regularizers.l2(1e-4),
                                         name=name + '_1_bn')(x1)
    x1 = keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3,
                             padding='same',
                             use_bias=False,
                             name=name + '_2_conv')(x1)
    x = keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
