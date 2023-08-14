"""
Authors : inzapp

Github url : https://github.com/inzapp/denoising-model

Copyright (c) 2023 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, bn=False):
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='dn_input')
        x = input_layer
        x = self.conv2d(x, 4, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 4, 3, 1, bn=bn, activation='leaky')
        output_layer = self.denoising_layer(x, name='dn_output')
        return tf.keras.models.Model(input_layer, output_layer)

    def denoising_layer(self, x, name):
        x = tf.keras.layers.Conv2D(
            filters=self.input_shape[-1],
            padding='same',
            kernel_size=1,
            kernel_initializer=self.kernel_initializer())(x)
        return self.activation(x, 'sigmoid', name=name)

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def activation(self, x, activation, name=None):
        if activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x

