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
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 image_paths_gt,
                 image_paths_noisy,
                 user_input_shape,
                 model_input_shape,
                 input_type,
                 batch_size,
                 max_noise,
                 dtype='float32'):
        assert input_type in ['gray', 'rgb', 'nv12', 'nv21']
        self.image_paths_gt = image_paths_gt
        self.image_paths_noisy = image_paths_noisy
        self.user_input_shape = user_input_shape
        self.model_input_shape = model_input_shape
        self.input_type = input_type
        self.batch_size = batch_size
        self.max_noise = max_noise
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        self.img_size = (self.user_input_shape[1], self.user_input_shape[0])
        self.noisy_image_paths_of = self.get_noisy_image_paths_of(self.image_paths_gt, self.image_paths_noisy)
        np.random.shuffle(image_paths_gt)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])
        self.transform_noise = A.Compose([
            A.ImageCompression(p=0.5, quality_lower=10, quality_upper=30),
            A.GaussNoise(p=1.0, var_limit=(0.0, 100.0)),
        ])

    def __len__(self):
        return int(np.floor(len(self.image_paths_gt) / self.batch_size))

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x, batch_y = [], []
        for f in fs:
            img, img_noisy = f.result()
            if self.input_type in ['nv12', 'nv21']:
                img = self.convert_bgr2yuv420sp(img)
                img_noisy = self.convert_bgr2yuv420sp(img_noisy)
            elif self.input_type == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_x.append(self.normalize(np.asarray(img_noisy).reshape(self.model_input_shape)))
            batch_y.append(self.normalize(np.asarray(img).reshape(self.model_input_shape)))
        batch_x = np.asarray(batch_x).astype(self.dtype)
        batch_y = np.asarray(batch_y).astype(self.dtype)
        return batch_x, batch_y

    def key(self, image_path, image_type):
        if image_type == 'gt':
            key = f'{os.path.basename(image_path)[:-4]}'
        else:
            key = f'{os.path.basename(image_path)[:-12]}'
        return key

    def get_noisy_image_paths_of(self, image_paths_gt, image_paths_noisy):
        noisy_image_paths_of = {}
        for path in image_paths_gt:
            noisy_image_paths_of[self.key(path, 'gt')] = []
        for path in image_paths_noisy:
            noisy_image_paths_of[self.key(path, 'noisy')].append(path)
        return noisy_image_paths_of

    def normalize(self, x):
        return np.clip(np.asarray(x).astype('float32') / 255.0, 0.0, 1.0)

    def denormalize(self, x):
        return np.asarray(np.clip((x * 255.0), 0.0, 255.0)).astype('uint8')

    def next_image_path(self):
        path = self.image_paths_gt[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths_gt):
            self.img_index = 0
            np.random.shuffle(self.image_paths_gt)
        return path

    def resize(self, img, size=(-1, -1), scale=1.0):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if scale != 1.0:
            if scale > 1.0:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, (0, 0), fx=scale, fy=scale)
        else:
            if size[0] > img_width or size[1] > img_height:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, size, interpolation=interpolation)

    def convert_bgr2yuv420sp(self, img):
        assert self.input_type in ['nv12', 'nv21']
        h, w = img.shape[:2]
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_YV12)
        y = yuv[:h]
        uv = yuv[h:]
        uvf = uv.flatten()
        v = uvf[:int(uvf.shape[0] / 2)].reshape(uv.shape[0], -1)
        u = uvf[int(uvf.shape[0] / 2):].reshape(uv.shape[0], -1)
        new_uv_or_vu = np.zeros(uv.shape, dtype=uv.dtype)
        if self.input_type == 'nv12':
            new_uv_or_vu[:,::2] = u
            new_uv_or_vu[:,1::2] = v
        elif self.input_type == 'nv21':
            new_uv_or_vu[:,::2] = v
            new_uv_or_vu[:,1::2] = u
        return np.vstack((y, new_uv_or_vu))

    def convert_yuv420sp2bgr(self, img):
        assert self.input_type in ['nv12', 'nv21']
        if self.input_type == 'nv12':
            conversion_type = cv2.COLOR_YUV2BGR_NV12
        else:
            conversion_type = cv2.COLOR_YUV2BGR_NV21
        return cv2.cvtColor(img, conversion_type)

    def add_noise(self, img):
        img_noisy = np.array(img).astype('float32')
        noise_power = np.random.uniform() * self.max_noise
        img_noisy += np.random.uniform(-noise_power, noise_power, size=img.shape)
        img_noisy = np.clip(img_noisy, 0.0, 255.0).astype('uint8')
        return img_noisy

    def noisy_image_path(self, image_path):
        return np.random.choice(self.noisy_image_paths_of[self.key(image_path, 'gt')])

    def load(self, image_path):
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if self.input_type == 'gray' else cv2.IMREAD_COLOR)
        img = self.resize(img, self.img_size)
        return img

    def load_gt_image(self, image_path_gt):
        return self.load(image_path_gt)

    def load_noisy_image(self, image_path_gt):
        return self.load(self.noisy_image_path(image_path_gt))

    def load_image(self, image_path_gt):
        img = self.load_gt_image(image_path_gt)
        if np.random.uniform() < 0.2:
            img_noisy = img.copy()
        else:
            img_noisy = self.load_noisy_image(image_path_gt)
        return img, img_noisy

