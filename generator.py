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
import cv2
import numpy as np
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 image_paths,
                 input_shape,
                 input_type,
                 batch_size,
                 max_noise,
                 dtype='float32'):
        assert input_type in ['gray', 'rgb', 'nv12', 'nv21']
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.input_type = input_type
        self.batch_size = batch_size
        self.max_noise = max_noise
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(image_paths)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])
        self.transform_noise = A.Compose([
            A.ImageCompression(p=0.5, quality_lower=75),
            A.ISONoise(p=0.5, color_shift=(0.01, 0.05), intensity=(0.1, 0.75)),  # must be rgb image
            A.GaussNoise(p=0.5, var_limit=(100.0, 100.0)),
            A.MultiplicativeNoise(p=0.5, multiplier=(0.8, 1.2), elementwise=True)
        ])
        if input_type in ['nv12', 'nv21']:
            self.yuv_mask = self.make_yuv_mask()
            self.num_yuv_pos = np.sum(self.yuv_mask)
        else:
            self.yuv_mask = 1.0
            self.num_yuv_pos = 1.0

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def make_yuv_mask(self):
        target_shape = (self.batch_size,) + self.input_shape
        mask = np.ones(shape=target_shape, dtype=np.float32)
        input_rows = self.input_shape[0]
        mask[:, (input_rows // 2):, :, 1] = 0.0
        mask[:, :, :, 2] = 0.0
        return mask

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x, batch_y = [], []
        for f in fs:
            img, img_noise = f.result()
            batch_x.append(self.normalize(np.asarray(img_noise).reshape(self.input_shape)))
            batch_y.append(self.normalize(np.asarray(img).reshape(self.input_shape)))
        batch_x = np.asarray(batch_x).astype(self.dtype)
        batch_y = np.asarray(batch_y).astype(self.dtype)
        return batch_x, batch_y, self.yuv_mask, self.num_yuv_pos

    @staticmethod
    def normalize(x):
        return np.clip(np.asarray(x).astype('float32') / 255.0, 0.0, 1.0)

    @staticmethod
    def denormalize(x):
        return np.asarray(np.clip((x * 255.0), 0.0, 255.0)).astype('uint8')

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def resize(self, img, size):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if size[0] > img_width or size[1] > img_height:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        return cv2.resize(img, size, interpolation=interpolation)

    def convert_bgr2yuv3ch(self, img, yuv_type):
        assert yuv_type in ['nv12', 'nv21']
        h, w, c = img.shape
        new_img = np.zeros(shape=img.shape, dtype=np.uint8)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_YV12)
        y = yuv[:h]
        uv = yuv[h:]
        uvf = uv.flatten()
        v = uvf[:int(uvf.shape[0] / 2)].reshape(uv.shape[0], -1)
        u = uvf[int(uvf.shape[0] / 2):].reshape(uv.shape[0], -1)
        new_uv = np.zeros(uv.shape)
        if yuv_type == 'nv12':
            new_uv[:,::2] = u
            new_uv[:,1::2] = v
        elif yuv_type == 'nv21':
            new_uv[:,::2] = v
            new_uv[:,1::2] = u
        new_uv_zero_padded = np.vstack((new_uv, np.zeros(shape=new_uv.shape, dtype=np.uint8)))
        new_yuv = np.zeros(shape=img.shape, dtype=np.uint8)
        new_yuv[:, :, 0] = y
        new_yuv[:, :, 1] = new_uv_zero_padded
        return new_yuv

    def convert_yuv3ch2bgr(self, yuv, yuv_type):
        h, w, c = yuv.shape
        y = yuv[:, :, 0]
        uv_or_vu = yuv[:, :, 1][0:h//2, 0:w]
        yuv = np.vstack((y, uv_or_vu))
        if yuv_type == 'nv12':
            bgr = np.asarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12))
        else:
            bgr = np.asarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21))
        return bgr

    def load_image(self, image_path):
        if np.random.uniform() < 0.0:
            background_color = np.random.uniform(size=self.input_shape[-1]) * 0.25
            img = (background_color.astype('float32').reshape((1, 1, self.input_shape[-1])) * 255.0).astype('uint8')
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)

            foreground_color = np.random.uniform(size=3) * 255.0 * 0.25
            x1 = int(np.random.uniform() * self.input_shape[1])
            y1 = int(np.random.uniform() * self.input_shape[0])
            w = int(np.random.uniform() * self.input_shape[1])
            h = int(np.random.uniform() * self.input_shape[0])
            if np.random.uniform() < 0.5:
                x2 = x1 + w
                y2 = y1 + h
                img = cv2.rectangle(img, (x1, y1), (x2, y2), foreground_color, -1)
            else:
                cx = int(x1 + (w * 0.5))
                cy = int(y1 + (h * 0.5))
                radius = w
                img = cv2.circle(img, (cx, cy), radius, foreground_color, -1)
        else:
            data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if self.input_type == 'gray' else cv2.IMREAD_COLOR)
            img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
            if self.input_type == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_type in ['nv12', 'nv21']:
            img = self.convert_bgr2yuv3ch(img, self.input_type)

        img_noise = np.array(img).astype('float32')
        noise_power = np.random.uniform() * self.max_noise
        img_noise += np.random.uniform(-noise_power, noise_power, size=img.shape)
        img_noise = np.clip(img_noise, 0.0, 255.0).astype('uint8')
        return img, img_noise

    # def load_image(self, image_path):
    #     if self.denoising_model or self.input_shape[-1] == 3:
    #         img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # ISONoise need rgb image
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     else:
    #         img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #     img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
    #     img = self.transform(image=img)['image']
    #     img_noise = None
    #     if self.denoising_model:
    #         img_noise = self.transform_noise(image=img)['image']
    #         if self.input_shape[-1] == 1:
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #             img_noise = cv2.cvtColor(img_noise, cv2.COLOR_RGB2GRAY)
    #     return img, img_noise

