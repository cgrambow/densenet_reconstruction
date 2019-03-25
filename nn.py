#!/usr/bin/env python
# -*- coding:utf-8

import os

import keras

import layers

K = keras.backend


def build(input_shape=(256, 256, 1), blocks=3, growth_rate=12, residual_connections=True):
    channels = (8, 16, 32, 64, 128)
    encoder = []  # Store encoder tensors for residual connections

    inputs = keras.layers.Input(shape=input_shape)

    # Encoder
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = keras.layers.Conv2D(channels[0], 7, use_bias=False, name='down/conv1/conv')(x)
    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='down/conv1/bn')(x)
    x = keras.layers.Activation('relu', name='down/conv1/relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name='down/pool1')(x)
    encoder.append(x)

    for i, c in enumerate(channels[1:]):
        x = layers.dense_block(x, blocks, growth_rate, name='down/conv' + str(i + 2))
        x = layers.transition_block_down(x, c, name='down/pool' + str(i + 2))
        encoder.append(x)

    # Decoder
    i = 0
    for i, c in enumerate(channels[-2::-1]):
        x = layers.dense_block(x, blocks, growth_rate, name='up/conv' + str(i + 1))
        x = layers.transition_block_up(x, c, name='up/up' + str(i + 1))
        if residual_connections:
            x = keras.layers.add([x, encoder[-(i + 2)]], name='up/skip' + str(i + 1))

    x = layers.dense_block(x, blocks, growth_rate, name='up/conv' + str(i + 2))
    x = layers.transition_block_up(x, 4, name='up/up' + str(i + 2))
    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='up/conv{}/bn_2'.format(i + 2))(x)
    x = keras.layers.Activation('relu', name='up/conv{}/relu_2'.format(i + 2))(x)

    # Final layer
    x = keras.layers.ZeroPadding2D(padding=((2, 2), (2, 2)))(x)
    x = keras.layers.Conv2D(1, 5, use_bias=False, name='last/conv')(x)
    x = keras.layers.BatchNormalization(axis=layers.bn_axis, epsilon=1.001e-5,
                                        gamma_regularizer=keras.layers.regularizers.l2(1e-4),
                                        name='last/bn')(x)
    x = keras.layers.Activation('sigmoid', name='last/sigmoid')(x)

    # Return model
    return keras.models.Model(inputs, x, name='dense_unet')


def train(model, x, y, save_dir,
          batch_size=16, max_epochs=100, validation_split=0.05, patience=10):
    model.compile(loss=npcc, optimizer='adam')
    model.summary()

    model_name = 'model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 verbose=1,
                                                 save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=patience,
                                                   verbose=1,
                                                   restore_best_weights=True)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                   patience=5,
                                                   min_lr=1e-6)
    callbacks = [checkpoint, early_stopping, lr_reducer]

    model.fit(x, y,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_split=validation_split,
              shuffle=True,
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
