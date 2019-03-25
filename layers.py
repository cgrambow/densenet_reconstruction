#!/usr/bin/env python
# -*- coding:utf-8

import keras

bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1


def dense_block(x, blocks, growth_rate, bottleneck, name, output_last=False):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        bottleneck: bool, use 1x1 bottleneck convolution
        name: string, block label.
        output_last: bool, return last dense block output
    # Returns
        output tensor for the block.
    """
    last = []
    for i in range(blocks):
        x1 = conv_block(x, growth_rate, bottleneck, name=name + '/block' + str(i + 1))
        last.append(x1)
        x = keras.layers.Concatenate(axis=bn_axis, name=name + '/concat' + str(i + 1))([x, x1])
    if output_last:
        xlast = keras.layers.Concatenate(axis=bn_axis, name=name + '/concat_last')(last)
        return xlast
    return x


def transition_block_down(x, reduction, name):
    """A downsampling transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate for filters.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.regularizers.l2(1e-4),
                                        name=name + '/bn')(x)
    x = keras.layers.Activation('relu', name=name + '/relu')(x)
    x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[bn_axis] * reduction), 1,
                            use_bias=False,
                            kernel_initializer=keras.initializers.he_uniform(),
                            name=name + '/conv')(x)
    x = keras.layers.Dropout(0.2, name=name + '/dropout')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '/pool')(x)
    return x


def transition_block_up(x, xskip, filters, name):
    """An upsampling transition block.
    # Arguments
        x: input tensor.
        xskip: skip connection.
        filters: int, number of filters in convolutional layer.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = keras.layers.Conv2DTranspose(filters, 3, strides=2,
                                     padding='same',
                                     use_bias=False,
                                     kernel_initializer=keras.initializers.he_uniform(),
                                     name=name + '/upconv')(x)
    x = keras.layers.Concatenate(axis=bn_axis, name=name + '/skip')([x, xskip])
    return x


def conv_block(x, growth_rate, bottleneck, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        bottleneck: bool, use 1x1 bottleneck convolution
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    if bottleneck:
        x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                            gamma_regularizer=keras.regularizers.l2(1e-4),
                                            name=name + '_0_bn')(x)
        x = keras.layers.Activation('relu', name=name + '_0_relu')(x)
        x = keras.layers.Conv2D(4 * growth_rate, 1,
                                use_bias=False,
                                kernel_initializer=keras.initializers.he_uniform(),
                                name=name + '_0_conv')(x)
        x = keras.layers.Dropout(0.2, name=name + '_0_dropout')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.regularizers.l2(1e-4),
                                        name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.Conv2D(growth_rate, 3,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=keras.initializers.he_uniform(),
                            name=name + '_conv')(x)
    x = keras.layers.Dropout(0.2, name=name + '_dropout')(x)
    return x
