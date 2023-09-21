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
        assert self.input_shape[0] % 4 == 0 and self.input_shape[1] % 4 == 0
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='dn_input')
        x = input_layer
        x = self.conv2d(x, 16, 3, 1, bn=bn, activation='leaky')
        f0 = x
        x = self.maxpool2d(x)
        x = self.conv2d(x, 32, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 32, 3, 1, bn=bn, activation='leaky')
        f1 = x
        x = self.maxpool2d(x)
        x = self.conv2d(x, 64, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 64, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 64, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 32, 1, 1, bn=bn, activation='leaky')
        x = self.upsampling2d(x)
        x = self.add([x, f1])
        x = self.conv2d(x, 32, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 32, 3, 1, bn=bn, activation='leaky')
        x = self.conv2d(x, 16, 1, 1, bn=bn, activation='leaky')
        x = self.upsampling2d(x)
        x = self.add([x, f0])
        x = self.conv2d(x, 16, 3, 1, bn=bn, activation='leaky')
        output_layer = self.denoising_layer(x, input_layer, name='dn_output')
        return tf.keras.models.Model(input_layer, output_layer)

    def denoising_layer(self, x, input_layer, name='dn_output'):
        x = self.conv2d(x, self.input_shape[-1], 1, 1, bn=bn, activation='tanh')
        return self.add([x, input_layer], name=name)

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def maxpool2d(self, x):
        return tf.keras.layers.MaxPooling2D()(x)

    def upsampling2d(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, x, name=None):
        return tf.keras.layers.Add(name=name)(x)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(l2=0.01)

    def activation(self, x, activation, name=None):
        if activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x

