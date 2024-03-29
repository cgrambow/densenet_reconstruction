#!/usr/bin/env python
# -*- coding:utf-8

import math
import os

import keras

import layers

K = keras.backend


def build(input_shape=(256, 256, 1),
          feat_maps=16, growth_rate=12, blocks=(3, 5, 7, 9, 11),
          dropout=0.2, reduction=0.5, bottleneck=True):
    """Build dense u-net.
    # Arguments
        input_shape: shape of input tensor.
        feat_maps: int, number of feature maps in first convolutional layer.
        growth_rate: int, growth rate in dense blocks.
        blocks: number of layers in each dense block.
        dropout: float, dropout rate after convolutions.
        reduction: float, compression rate for filters in transition blocks.
        bottleneck: bool, use 1x1 bottleneck convolution in dense blocks.
    # Returns
        model
    """
    encoder = []  # Store encoder tensors for skip connections

    inputs = keras.layers.Input(shape=input_shape)

    # Encoder
    x = keras.layers.Conv2D(feat_maps, 7,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=keras.initializers.he_uniform(),
                            name='down/conv1/conv')(inputs)

    for i, b in enumerate(blocks[:-1]):
        x = layers.dense_block(x, b,
                               growth_rate=growth_rate,
                               dropout=dropout,
                               bottleneck=bottleneck,
                               name='down/dense' + str(i + 2))
        encoder.append(x)
        feat_maps += b * growth_rate
        x = layers.transition_block_down(x, dropout=dropout,
                                         reduction=reduction,
                                         name='down/tr' + str(i + 2))
    encoder.reverse()

    # Central layers
    x = layers.dense_block(x, blocks[-1],
                           growth_rate=growth_rate,
                           dropout=dropout,
                           bottleneck=bottleneck,
                           output_last=True,
                           name='middle/dense')
    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='middle/bn')(x)
    x = keras.layers.Activation('relu', name='middle/relu')(x)

    # Decoder
    blocks = blocks[::-1]
    i = 0
    for i, b in enumerate(blocks[1:]):
        feat_maps = int(blocks[i] * growth_rate * reduction)
        x = layers.transition_block_up(x, encoder[i], feat_maps, name='up/tr' + str(i + 1))
        x = layers.dense_block(x, b,
                               growth_rate=growth_rate,
                               dropout=dropout,
                               bottleneck=bottleneck,
                               output_last=(i + 1 < len(blocks[1:])),
                               name='up/dense' + str(i + 1))

    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='up/conv{}/bn_2'.format(i + 2))(x)
    x = keras.layers.Activation('relu', name='up/conv{}/relu_2'.format(i + 2))(x)

    # Final layer
    x = keras.layers.Conv2D(1, 5,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=keras.initializers.glorot_uniform(),
                            name='last/conv')(x)
    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='last/bn')(x)
    x = keras.layers.Activation('sigmoid', name='last/sigmoid')(x)

    # Return model
    return keras.models.Model(inputs, x, name='dense_unet')


def train(model, x, y, save_dir,
          batch_size=32, max_epochs=500, validation_split=0.05, patience=10,
          lr0=0.001, decay=0.1, maxnorm=1.0):
    optimizer = keras.optimizers.Adam(lr=lr0, clipnorm=maxnorm)
    model.compile(loss=npcc, optimizer=optimizer)
    model.summary()

    model_name = 'model.{epoch:03d}.h5'
    model_path = os.path.join(save_dir, model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 verbose=1,
                                                 save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=patience,
                                                   verbose=1,
                                                   restore_best_weights=True)
    lr_scheduler = keras.callbacks.LearningRateScheduler(lambda e, lr: lr0 * math.exp(-decay * e),
                                                         verbose=1)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                   patience=5)
    callbacks = [checkpoint, early_stopping, lr_scheduler, lr_reducer]

    model.fit(x, y,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_split=validation_split,
              shuffle=True,
              verbose=2,
              callbacks=callbacks)


def npcc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = r_num / r_den
    return -r
