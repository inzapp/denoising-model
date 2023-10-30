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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import random
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
from ckpt_manager import CheckpointManager


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
        self.batch_size = batch_size
        self.iterations = iterations
        self.save_interval = save_interval
        self.training_view = training_view


class DenoisingModel(CheckpointManager):
    def __init__(self, config, training):
        assert config.save_interval >= 1000
        self.pretrained_model_path = config.pretrained_model_path
        self.train_image_path = config.train_image_path
        self.validation_image_path = config.validation_image_path
        self.input_type = config.input_type
        self.user_input_shape = (config.input_rows, config.input_cols, 3 if config.input_type == 'rgb' else 1)
        self.model_input_shape = self.convert_to_model_input_shape(self.user_input_shape)
        self.lr = config.lr
        self.warm_up = config.warm_up
        self.batch_size = config.batch_size
        self.save_interval = config.save_interval
        self.iterations = config.iterations
        self.training_view = config.training_view
        warnings.filterwarnings(action='ignore')
        self.set_model_name(config.model_name)

        self.live_view_previous_time = time()
        if not training:
            self.set_global_seed()

        if not self.is_valid_path(self.train_image_path):
            print(f'train image path is not valid : {self.train_image_path}')
            exit(0)

        if not self.is_valid_path(self.validation_image_path):
            print(f'validation image path is not valid : {self.validation_image_path}')
            exit(0)

        self.train_image_paths_gt, self.train_image_paths_noisy = self.init_image_paths(self.train_image_path)
        if len(self.train_image_paths_gt) == 0:
            print(f'no images found in {self.train_image_path}')
            exit(0)
        if len(self.train_image_paths_noisy) == 0:
            print(f'no noisy images found in {self.train_image_path}')
            exit(0)
        self.validation_image_paths_gt, self.validation_image_paths_noisy = self.init_image_paths(self.validation_image_path)
        if len(self.validation_image_paths_gt) == 0:
            print(f'no images found in {self.validation_image_path}')
            exit(0)
        if len(self.validation_image_paths_noisy) == 0:
            print(f'no noisy images found in {self.validation_image_path}')
            exit(0)

        self.pretrained_iteration_count = 0
        if self.pretrained_model_path != '':
            self.model, self.user_input_shape, self.model_input_shape = self.load_model(self.pretrained_model_path)
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.pretrained_model_path)
        else:
            self.model = Model(input_shape=self.model_input_shape).build()

        self.data_generator = DataGenerator(
            image_paths_gt=self.train_image_paths_gt,
            image_paths_noisy=self.train_image_paths_noisy,
            user_input_shape=self.user_input_shape,
            model_input_shape=self.model_input_shape,
            input_type=self.input_type,
            batch_size=self.batch_size)

    def convert_to_model_input_shape(self, user_input_shape):
        if self.input_type in ['nv12', 'nv21']:
            return (int(user_input_shape[0] * 1.5), user_input_shape[1], user_input_shape[2])
        else:
            return user_input_shape

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def init_image_paths(self, image_path):
        paths_all = glob(f'{image_path}/**/*.jpg', recursive=True)
        paths_gt, paths_noisy = [], []
        for path in paths_all:
            if os.path.basename(path).find('_NOISY_') == -1:
                paths_gt.append(path)
            else:
                paths_noisy.append(path)
        return paths_gt, paths_noisy

    def load_model(self, model_path):
        import tensorflow as tf
        if not (os.path.exists(model_path) and os.path.isfile(model_path)):
            print(f'file not found : {model_path}')
            exit(0)
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf})
        model_input_shape = model.input_shape[1:]
        if self.input_type in ['nv12', 'nv21']:
            user_input_shape = (model_input_shape[0] // 3 * 2, model_input_shape[1], model_input_shape[2])
        else:
            user_input_shape = model_input_shape
        return model, user_input_shape, model_input_shape

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(mse, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mse

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def concat(self, images):
        return np.concatenate(images, axis=1)

    def non_local_means_filter(self, img, h=10, template_window_size=7, search_window_size=21):
        img = img.reshape(self.model_input_shape)
        if img.shape[-1] == 1:
            img_denoised = cv2.fastNlMeansDenoising(img, None, h, template_window_size, search_window_size)
        else:
            img_denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window_size, search_window_size)
        return img_denoised

    # input image : gray or bgr image, output image : denoised gray or bgr image for viewing
    def predict(self, img_noisy, model_forward=True):
        if self.input_type in ['nv12', 'nv21']:
            img_noisy = self.data_generator.convert_bgr2yuv420sp(img_noisy)
        elif self.input_type == 'rgb':
            img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB)

        if model_forward:
            x = self.data_generator.normalize(img_noisy).reshape((1,) + self.model_input_shape)
            y = np.array(self.graph_forward(self.model, x)).reshape(self.model_input_shape)
            img_denoised = self.data_generator.denormalize(y)
        else:
            img_denoised = img_noisy.copy()

        if self.input_type in ['nv12', 'nv21']:
            img_noisy = self.data_generator.convert_yuv420sp2bgr(img_noisy)
            img_denoised = self.data_generator.convert_yuv420sp2bgr(img_denoised)
        elif self.input_type == 'rgb':
            img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_RGB2BGR)
            img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2BGR)

        view_shape = self.user_input_shape[:2] + (1 if self.input_type == 'gray' else 3,)
        img_noisy = img_noisy.reshape(view_shape)
        img_denoised = img_denoised.reshape(view_shape)
        return img_noisy, img_denoised

    def predict_images(self, image_path='', dataset='validation', save_count=0, predict_gt=False):
        image_paths_gt, image_paths_noisy = [], []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths_gt, image_paths_noisy = self.init_image_paths(image_path)
            else:
                image_paths_gt, image_paths_noisy = [image_path], []
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths_gt = self.train_image_paths_gt
                image_paths_noisy = self.train_image_paths_noisy
            else:
                image_paths_gt = self.validation_image_paths_gt
                image_paths_noisy = self.validation_image_paths_noisy

        if len(image_paths_gt) == 0:
            print(f'no images found')
            return

        data_generator = DataGenerator(
            image_paths_gt=image_paths_gt,
            image_paths_noisy=image_paths_noisy,
            user_input_shape=self.user_input_shape,
            model_input_shape=self.model_input_shape,
            input_type=self.input_type,
            batch_size=self.batch_size)

        cnt = 0
        save_path = 'result_images'
        os.makedirs(save_path, exist_ok=True)
        for path in image_paths_gt:
            img_noisy = data_generator.load_gt_image(path) if predict_gt else data_generator.load_noisy_image(path)
            img_noisy, img_denoised = self.predict(img_noisy)
            img_concat = self.concat([img_noisy, img_denoised])
            if save_count > 0:
                basename = os.path.basename(path)
                save_img_path = f'{save_path}/{basename}'
                cv2.imwrite(save_img_path, img_concat, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cnt += 1
                print(f'[{cnt} / {save_count}] save success : {save_img_path}')
                if cnt == save_count:
                    break
            else:
                cv2.imshow('img_denoised', img_concat)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)

    def predict_video(self, video_path):
        if not (os.path.exists(video_path) and os.path.isfile(video_path)):
            print(f'video not found. video video_path : {video_path}')
            exit(0)
        cap = cv2.VideoCapture(video_path)
        while True:
            frame_exist, bgr = cap.read()
            if not frame_exist:
                print('frame not exists')
                break
            bgr = self.data_generator.resize(bgr, (self.user_input_shape[1], self.user_input_shape[0]))
            img_noisy = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if self.input_type == 'gray' else bgr
            img_noisy, img_denoised = self.predict(img_noisy)
            img_concat = self.concat([img_noisy, img_denoised])
            img_concat = self.data_generator.resize(img_concat, scale=0.5)
            cv2.imshow('img_denoised', img_concat)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cap.release()
        cv2.destroyAllWindows()

    def predict_rtsp(self, rtsp_url):
        import threading
        from time import sleep
        def read_frames(rtsp_url, frame_queue, end_flag, lock):
            cap = cv2.VideoCapture(rtsp_url)
            while True:
                with lock:
                    frame_exist, bgr = cap.read()
                    if not frame_exist:
                        break
                    if len(frame_queue) == 0:
                        frame_queue.append(bgr)
                    else:
                        frame_queue[0] = bgr
                    print(f'[read_frames] frame updated')
                sleep(0)
            end_flag[0] = True
            cap.release()

        lock, frame_queue, end_flag = threading.Lock(), [], [False]
        read_thread = threading.Thread(target=read_frames, args=(rtsp_url, frame_queue, end_flag, lock))
        read_thread.daemon = True
        read_thread.start()
        while True:
            if end_flag[0]:
                print(f'[main] end flag is True')
                break
            bgr = None
            with lock:
                if frame_queue:
                    bgr = frame_queue[0].copy()
            if bgr is None:
                print(f'[main] bgr is None')
                sleep(0.1)
                continue

            print(f'[main] frame get success')
            bgr = self.data_generator.resize(bgr, (self.user_input_shape[1], self.user_input_shape[0]))
            img_noisy = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if self.input_type == 'gray' else bgr
            img_noisy, img_denoised = self.predict(img_noisy)
            img_concat = self.concat([img_noisy, img_denoised])
            img_concat = self.data_generator.resize(img_concat, scale=0.5)
            cv2.imshow('img_denoised', img_concat)
            key = cv2.waitKey(1)
            if key == 27:
                exit(0)
        cv2.destroyAllWindows()

    def psnr(self, mse):
        return 20 * np.log10(1.0 / np.sqrt(mse)) if mse!= 0.0 else 100.0

    def evaluate(self, dataset='validation', image_path='', evaluate_gt=False):
        image_paths_gt, image_paths_noisy = [], []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths_gt, image_paths_noisy = self.init_image_paths(image_path)
            else:
                image_paths_gt, image_paths_noisy = [image_path], []
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths_gt = self.train_image_paths_gt
                image_paths_noisy = self.train_image_paths_noisy
            else:
                image_paths_gt = self.validation_image_paths_gt
                image_paths_noisy = self.validation_image_paths_noisy

        if len(image_paths_gt) == 0:
            print(f'no images found')
            return

        data_generator = DataGenerator(
            image_paths_gt=image_paths_gt,
            image_paths_noisy=image_paths_noisy,
            user_input_shape=self.user_input_shape,
            model_input_shape=self.model_input_shape,
            input_type=self.input_type,
            batch_size=self.batch_size)

        psnr_sum = 0.0
        ssim_sum = 0.0
        for path in tqdm(image_paths_gt):
            img = data_generator.load_gt_image(path)
            img_noisy = data_generator.load_noisy_image(path)

            img, _ = self.predict(img, model_forward=False)  # for convert to gray or bgr
            img_noisy, img_denoised = self.predict(img_noisy, model_forward=not evaluate_gt)

            img_true_norm = data_generator.normalize(img)
            img_pred_norm = data_generator.normalize(img_denoised)
            mse = np.mean((img_true_norm - img_pred_norm) ** 2.0)
            ssim = tf.image.ssim(img_true_norm, img_pred_norm, 1.0)
            psnr = self.psnr(mse)
            psnr_sum += psnr
            ssim_sum += ssim
        avg_psnr = psnr_sum / float(len(image_paths_gt))
        avg_ssim = ssim_sum / float(len(image_paths_gt))
        print(f'\npsnr : {avg_psnr:.2f}, ssim : {avg_ssim:.4f}')

    def train(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths_gt)} gt, {len(self.train_image_paths_noisy)} noisy samples.')
        print('start training')
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        is_yuv = self.input_type in ['nv12', 'nv21']
        self.init_checkpoint_dir()
        print(f'checkpoint path : {self.checkpoint_path}')
        while True:
            for batch_x, batch_y in self.data_generator:
                lr_scheduler.update(optimizer, iteration_count)
                mse = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
                iteration_count += 1
                print(f'\r[iteration_count : {iteration_count:6d}] loss : {mse:>8.4f}, psnr : {self.psnr(mse):.2f}', end='')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 2000 == 0:
                    self.save_last_model(self.model, iteration_count)
                if iteration_count == self.iterations:
                    print('\ntrain end successfully')
                    return

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            img_path = np.random.choice(self.validation_image_paths_gt)
            img, img_noisy = self.data_generator.load_image(img_path)
            img_denoised = self.predict(img_noisy)
            if self.input_type in ['nv12', 'nv21']:
                img_noisy = self.data_generator.convert_yuv420sp2bgr(img_noisy)
            img_concat = self.concat([img_noisy, img_denoised])
            cv2.imshow('training view', img_concat)
            cv2.waitKey(1)

