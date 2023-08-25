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
import warnings
import numpy as np
import shutil as sh
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from generator import DataGenerator
from lr_scheduler import LRScheduler


class TrainingConfig:
    def __init__(self,
                 pretrained_model_path='',
                 train_image_path='',
                 validation_image_path='',
                 input_rows=256,
                 input_cols=256,
                 input_type='gray',
                 model_name='model',
                 lr=0.001,
                 warm_up=0.1,
                 max_noise=30,
                 batch_size=2,
                 iterations=100000,
                 save_interval=5000,
                 training_view=False):
        self.pretrained_model_path = pretrained_model_path
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_type = input_type
        self.model_name = model_name
        self.lr = lr
        self.warm_up = warm_up
        self.max_noise = max_noise
        self.batch_size = batch_size
        self.iterations = iterations
        self.save_interval = save_interval
        self.training_view = training_view


class DenoisingModel:
    def __init__(self, config):
        assert config.save_interval >= 1000
        self.pretrained_model_path = config.pretrained_model_path
        self.train_image_path = config.train_image_path
        self.validation_image_path = config.validation_image_path
        self.input_shape = (config.input_rows, config.input_cols, 1 if config.input_type == 'gray' else 3)
        self.input_type = config.input_type
        self.model_name = config.model_name
        self.lr = config.lr
        self.warm_up = config.warm_up
        self.max_noise = config.max_noise
        self.batch_size = config.batch_size
        self.save_interval = config.save_interval
        self.iterations = config.iterations
        self.training_view = config.training_view
        warnings.filterwarnings(action='ignore')

        self.checkpoint_path = None
        self.pretrained_iteration_count = 0
        self.live_view_previous_time = time()

        if not self.is_valid_path(self.train_image_path):
            print(f'train image path is not valid : {self.train_image_path}')
            exit(0)

        if not self.is_valid_path(self.validation_image_path):
            print(f'validation image path is not valid : {self.validation_image_path}')
            exit(0)

        self.train_image_paths = self.init_image_paths(self.train_image_path)
        if len(self.train_image_paths) <= self.batch_size:
            print(f'image count({len(self.train_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)

        self.validation_image_paths = self.init_image_paths(self.validation_image_path)
        if len(self.validation_image_paths) <= self.batch_size:
            print(f'image count({len(self.validation_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)

        if self.pretrained_model_path != '':
            self.model, self.input_shape = self.load_model(self.pretrained_model_path)
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.pretrained_model_path)
        else:
            self.model = Model(input_shape=self.input_shape).build()
        self.data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            input_type=self.input_type,
            batch_size=self.batch_size,
            max_noise=self.max_noise)

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    def make_checkpoint_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def init_checkpoint_dir(self):
        inc = 0
        while True:
            if inc == 0:
                new_checkpoint_path = f'checkpoint/{self.model_name}'
            else:
                new_checkpoint_path = f'checkpoint/{self.model_name}_{inc}'
            if os.path.exists(new_checkpoint_path) and os.path.isdir(new_checkpoint_path):
                inc += 1
            else:
                break
        self.checkpoint_path = new_checkpoint_path
        self.make_checkpoint_dir()

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def load_model(self, model_path):
        if not (os.path.exists(model_path) and os.path.isfile(model_path)):
            print(f'file not found : {model_path}')
            exit(0)
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape[1:]
        return model, input_shape

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true, yuv_mask, num_yuv_pos, is_yuv):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.abs(y_true - y_pred)
            mse = tf.reduce_mean(tf.square(loss))
            ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
            if is_yuv:
                loss = tf.reduce_sum(loss) / (num_yuv_pos * tf.cast(tf.shape(x)[0], y_pred.dtype))
            else:
                loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, mse, ssim

    @tf.function
    def calculate_mse_ssim(self, model, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=False)
            loss = tf.abs(y_true - y_pred)
            mse = tf.reduce_mean(tf.square(loss))
            ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        return mse, ssim

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def predict(self, img, input_image_concat=True):
        if self.input_type in ['nv12', 'nv21']:
            origin_bgr = self.data_generator.convert_yuv3ch2bgr(img, self.input_type)
        elif self.input_type == 'rgb':
            origin_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            origin_bgr = img

        x = DataGenerator.normalize(img).reshape((1,) + self.input_shape)
        output = np.array(self.graph_forward(self.model, x)).reshape(self.input_shape)
        decoded_image = DataGenerator.denormalize(output)
        if self.input_type in ['nv12', 'nv21']:
            decoded_image = self.data_generator.convert_yuv3ch2bgr(decoded_image, self.input_type)
        elif self.input_type == 'rgb':
            decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)

        if input_image_concat:
            origin_bgr = origin_bgr.reshape(self.input_shape)
            decoded_image = decoded_image.reshape(self.input_shape)
            decoded_image = np.concatenate((origin_bgr, decoded_image), axis=1)
        return decoded_image

    def predict_images(self, image_path='', dataset='validation', save_count=0, recursive=False):
        image_paths = []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths = glob(f'{image_path}/**/*.jpg' if recursive else f'{image_path}/*.jpg', recursive=recursive)
            else:
                image_paths = [image_path]
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_image_paths
            else:
                image_paths = self.validation_image_paths

        if len(image_paths) == 0:
            print(f'no images found')
            return

        cnt = 0
        save_path = 'result_images'
        os.makedirs(save_path, exist_ok=True)
        for path in image_paths:
            _, img_noise = self.data_generator.load_image(path)
            decoded_image = self.predict(img_noise)
            if save_count > 0:
                basename = os.path.basename(path)
                save_img_path = f'{save_path}/{basename}'
                cv2.imwrite(save_img_path, decoded_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cnt += 1
                print(f'[{cnt} / {save_count}] save success : {save_img_path}')
                if cnt == save_count:
                    break
            else:
                cv2.imshow('decoded_image', decoded_image)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)

    def psnr(self, mse):
        return 20 * np.log10(1.0 / np.sqrt(mse)) if mse!= 0.0 else 100.0

    def evaluate(self, dataset='validation', image_path='', recursive=False):
        image_paths = []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths = glob(f'{image_path}/**/*.jpg' if recursive else f'{image_path}/*.jpg', recursive=recursive)
            else:
                image_paths = [image_path]
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_image_paths
            else:
                image_paths = self.validation_image_paths

        if len(image_paths) == 0:
            print(f'no images found')
            return

        np.random.seed(42)
        data_generator = DataGenerator(
            image_paths=image_paths,
            input_shape=self.input_shape,
            input_type=self.input_type,
            batch_size=1,
            max_noise=self.max_noise)

        cnt = 0
        psnr_sum = 0.0
        ssim_sum = 0.0
        for batch_x, batch_y, mask, num_pos in tqdm(data_generator):
            if self.input_type in ['nv12', 'nv21']:
                y = self.graph_forward(self.model, batch_x)
                bgr_true = data_generator.convert_yuv3ch2bgr(data_generator.denormalize(batch_x[0]), yuv_type=self.input_type)
                bgr_pred = data_generator.convert_yuv3ch2bgr(data_generator.denormalize(np.asarray(y[0])), yuv_type=self.input_type)
                bgr_true = data_generator.normalize(bgr_true)
                bgr_pred = data_generator.normalize(bgr_pred)
                mse = np.mean((bgr_true - bgr_pred) ** 2.0)
                ssim = tf.image.ssim(bgr_true, bgr_pred, 1.0)
            else:
                mse, ssim = self.calculate_mse_ssim(self.model, batch_x, batch_y)
            psnr_sum += self.psnr(mse)
            ssim_sum += ssim
            psnr = self.psnr(mse)
            cnt += 1
            if cnt == len(image_paths):
                break
        avg_psnr = psnr_sum / float(cnt)
        avg_ssim = ssim_sum / float(cnt)
        print(f'\nssim : {ssim:.4f}, psnr : {psnr:.2f}')

    def save_model(self, iteration_count):
        self.make_checkpoint_dir()
        save_path = f'{self.checkpoint_path}/model_{iteration_count}_iter.h5'
        self.model.save(save_path, include_optimizer=False)
        backup_path = f'{save_path}.bak'
        sh.move(save_path, backup_path)
        for last_model_path in glob(f'{self.checkpoint_path}/model_*_iter.h5'):
            os.remove(last_model_path)
        sh.move(backup_path, save_path)
        return save_path

    def train(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        is_yuv = self.input_type in ['nv12', 'nv21']
        self.init_checkpoint_dir()
        print(f'checkpoint path : {self.checkpoint_path}')
        while True:
            for batch_x, batch_y, mask, num_pos in self.data_generator:
                lr_scheduler.update(optimizer, iteration_count)
                loss, mse, ssim = self.compute_gradient(self.model, optimizer, batch_x, batch_y, mask, num_pos, is_yuv)
                iteration_count += 1
                print(f'\r[iteration_count : {iteration_count:6d}] loss : {loss:>8.4f}, ssim : {ssim:.4f}, psnr : {self.psnr(mse):.2f}', end='')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 2000 == 0:
                    self.save_model(iteration_count)
                if iteration_count == self.iterations:
                    print('\ntrain end successfully')
                    return

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            img_path = np.random.choice(self.validation_image_paths)
            img, img_noise = self.data_generator.load_image(img_path)
            decoded_image = self.predict(img_noise)
            cv2.imshow('training view', decoded_image)
            cv2.waitKey(1)

