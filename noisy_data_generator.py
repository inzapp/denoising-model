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
import argparse
import numpy as np
import shutil as sh
import albumentations as A

from glob import glob
from tqdm import tqdm


class NoisyDataGenerator:
    def __init__(self, generate_count_per_image):
        self.random_noise_max_range = 100
        self.transform_jpeg_compression = A.Compose([A.ImageCompression(quality_lower=30, quality_upper=50, always_apply=True)])
        self.generate_count_per_image = generate_count_per_image
        self.noise_functions = [
            self.random_noise_downscaled_1,
            self.random_noise_downscaled_2,
            self.random_noise_downscaled_4,
        ]
        self.min_noise_power_of = self.get_min_noise_power_of()
        self.max_noisy_image_count = 10  # fixed count for data loader, do not change
        self.noisy_index_candidates = list(map(str, list(range(self.max_noisy_image_count))))

    def get_min_noise_power_of(self):
        min_noise_power_of = {}
        min_noise_power_of[1] = 20
        min_noise_power_of[2] = 15
        min_noise_power_of[4] = 10
        return min_noise_power_of

    def init_image_paths(self, path):
        all_paths = glob(f'{path}/**/*.jpg', recursive=True)
        gt_paths = []
        noisy_paths = []
        for path in all_paths:
            if os.path.basename(path).find('_NOISY_') == -1:
                gt_paths.append(path)
            else:
                noisy_paths.append(path)
        return gt_paths, noisy_paths

    def add_random_noise(self, img, downscale_factor):
        img_noise = np.array(img).astype('float32')
        noise_power = np.random.uniform() * self.random_noise_max_range / downscale_factor
        noise_power = max(noise_power, self.min_noise_power_of[downscale_factor])
        img_h, img_w = img.shape[:2]
        noise = np.random.uniform(-noise_power, noise_power, size=(img_h // downscale_factor, img_w // downscale_factor, img.shape[2]))
        if downscale_factor > 1:
            noise = cv2.resize(noise, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        img_noise += noise
        img_noise = np.clip(img_noise, 0.0, 255.0).astype('uint8')
        return img_noise

    def random_noise_downscaled_1(self, img):
        return self.add_random_noise(img, downscale_factor=1)

    def random_noise_downscaled_2(self, img):
        return self.add_random_noise(img, downscale_factor=2)

    def random_noise_downscaled_4(self, img):
        return self.add_random_noise(img, downscale_factor=4)

    def jpeg_compression(self, img):
        return self.transform_jpeg_compression(image=img)['image']

    def add_noise(self, img):
        img = np.random.choice(self.noise_functions)(img)
        if np.random.uniform() < 0.5:
            img = self.jpeg_compression(img)
        return img

    def show(self, path):
        gt_paths, _ = self.init_image_paths(path)
        for path in gt_paths:
            print(path)
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_noise = self.add_noise(img)
            img_cat = np.concatenate([img, img_noise], axis=1)
            cv2.imshow('noisy_image_sample', img_cat)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)

    def generate(self, path):
        assert 1 <= self.generate_count_per_image <= self.max_noisy_image_count
        gt_paths, _ = self.init_image_paths(path)
        for path in tqdm(gt_paths):
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            path_without_extension = f'{path[:-4]}'
            save_count = 0
            while True:
                save_success = False
                img_noise = self.add_noise(img)
                for i in range(self.max_noisy_image_count):
                    save_path = f'{path_without_extension}_NOISY_{i}.jpg'
                    if not (os.path.exists(save_path) and os.path.isfile(save_path)):
                        cv2.imwrite(save_path, img_noise, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        save_count += 1
                        save_success = True
                        break
                if not save_success:  # case all noisy index image is generated
                    break
                if save_count == self.generate_count_per_image:
                    break

    def remove(self, path):
        _, noisy_paths = self.init_image_paths(path)
        if len(noisy_paths) == 0:
            print(f'no noisy images found in {path}')
            exit(0)
        for path in tqdm(noisy_paths):
            os.remove(path)

    def check(self, path):
        gt_paths, noisy_paths = self.init_image_paths(path)
        if len(gt_paths) == 0:
            print(f'no gt images found in {path}')
            exit(0)
        not_paired_gt_paths = []
        for gt_path in tqdm(gt_paths):
            noisy_image_exists = False
            for i in range(self.max_noisy_image_count):
                noisy_path = f'{gt_path[:-4]}_NOISY_{i}.jpg'
                if os.path.exists(noisy_path) and os.path.isfile(noisy_path):
                    noisy_image_exists = True
            if not noisy_image_exists:
                not_paired_gt_paths.append(gt_path)

        if len(not_paired_gt_paths) == 0:
            print('all images has noisy pairs at least once')
            print(f'gt images : {len(gt_paths)}')
            print(f'noisy images : {len(noisy_paths)}')
        else:
            for path in not_paired_gt_paths:
                print(path)
            print(f'\nno noisy pair image count : {len(not_paired_gt_paths)}')

    def rename(self, path):
        _, noisy_paths = self.init_image_paths(path)
        if len(noisy_paths) == 0:
            print(f'no noisy images found in {path}')
            exit(0)

        for path in tqdm(noisy_paths):
            path = path.replace('\\', '/')
            path_sp = path.split('/')
            basename = path_sp[-1]
            basename_without_extension = basename[:-4]
            basename_sp = basename_without_extension.split('_')

            noisy_position = -1
            noisy_index_position = -1
            for i in range(len(basename_sp)):
                if basename_sp[i] == 'NOISY' and i + 1 < len(basename_sp):
                    if basename_sp[i+1] in self.noisy_index_candidates:
                        noisy_position = i
                        noisy_index_position = i + 1
                        break

            new_basename = basename
            if noisy_position > -1 and noisy_index_position > -1:
                noisy_index_str = basename_sp.pop(noisy_index_position)
                basename_sp.pop(noisy_position)
                basename_sp.append('NOISY')
                basename_sp.append(noisy_index_str)
                new_basename = '_'.join(basename_sp) + '.jpg'
            path_sp[-1] = new_basename
            new_path = '/'.join(path_sp)
            if path != new_path:
                sh.move(path, new_path)

    def split(self, path):
        gt_paths, _= self.init_image_paths(path)
        if len(gt_paths) == 0:
            print(f'no gt images found in {path}')
            exit(0)

        train_dir_path = f'{path}/train'
        if not (os.path.exists(train_dir_path) and os.path.isdir(train_dir_path)):
            os.makedirs(train_dir_path, exist_ok=True)
        validation_dir_path = f'{path}/validation'
        if not (os.path.exists(validation_dir_path) and os.path.isdir(validation_dir_path)):
            os.makedirs(validation_dir_path, exist_ok=True)

        np.random.shuffle(gt_paths)

        validation_rate = 0.2
        validation_count = int(len(gt_paths) * validation_rate)
        train_gt_image_paths = gt_paths[validation_count:]
        validation_gt_image_paths = gt_paths[:validation_count]

        for path in tqdm(train_gt_image_paths):
            sh.move(path, train_dir_path)
            for index in self.noisy_index_candidates:
                noisy_path = f'{path[:-4]}_NOISY_{index}.jpg'
                if os.path.exists(noisy_path) and os.path.isfile(noisy_path):
                    sh.move(noisy_path, train_dir_path)
        for path in tqdm(validation_gt_image_paths):
            sh.move(path, validation_dir_path)
            for index in self.noisy_index_candidates:
                noisy_path = f'{path[:-4]}_NOISY_{index}.jpg'
                if os.path.exists(noisy_path) and os.path.isfile(noisy_path):
                    sh.move(noisy_path, validation_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='path where generate or remove noisy image')
    parser.add_argument('--count', type=int, default=1, help='generate count per image')
    parser.add_argument('--show', action='store_true', help='show noisy images for preview')
    parser.add_argument('--generate', action='store_true', help='generate noisy images')
    parser.add_argument('--remove', action='store_true', help='remove noisy images')
    parser.add_argument('--check', action='store_true', help='check dataset has paired noisy data')
    parser.add_argument('--rename', action='store_true', help='move tail identifier to end of name')
    parser.add_argument('--split', action='store_true', help='split train, validation dataset with noisy pairs')
    args = parser.parse_args()
    check_count = 0
    check_count = check_count + 1 if args.show else check_count
    check_count = check_count + 1 if args.generate else check_count
    check_count = check_count + 1 if args.remove else check_count
    check_count = check_count + 1 if args.rename else check_count
    check_count = check_count + 1 if args.check else check_count
    check_count = check_count + 1 if args.split else check_count
    if check_count == 0:
        print('use with one of [--show, --generate, --remove, --rename, --check, --split]')
        exit(0)
    if check_count > 1:
        print('[--show, --generate, --remove, --rename, --check, --split] cannot be used at the same time')
        exit(0)
    noisy_data_generator = NoisyDataGenerator(args.count)
    if args.show:
        noisy_data_generator.show(args.path)
    elif args.generate:
        noisy_data_generator.generate(args.path)
    elif args.remove:
        noisy_data_generator.remove(args.path)
    elif args.check:
        noisy_data_generator.check(args.path)
    elif args.rename:
        noisy_data_generator.rename(args.path)
    elif args.split:
        noisy_data_generator.split(args.path)

