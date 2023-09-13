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

from glob import glob
from tqdm import tqdm


class NoisyDataGenerator:
    def __init__(self, generate_count_per_image):
        self.generate_count_per_image = generate_count_per_image
        self.random_noise_max_range = 100

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

    def add_noise(self, img):
        img_noise = np.array(img).astype('float32')
        noise_power = np.random.uniform() * self.random_noise_max_range
        img_noise += np.random.uniform(-noise_power, noise_power, size=img.shape)
        img_noise = np.clip(img_noise, 0.0, 255.0).astype('uint8')
        return img_noise

    def generate(self, path):
        assert 1 <= self.generate_count_per_image <= 10
        gt_paths, _ = self.init_image_paths(path)
        for path in tqdm(gt_paths):
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            path_without_extension = f'{path[:-4]}'
            for i in range(self.generate_count_per_image):
                img_noise = self.add_noise(img)
                save_path = f'{path_without_extension}_NOISY_{i}.jpg'
                cv2.imwrite(save_path, img_noise, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def remove(self, path):
        _, noisy_paths = self.init_image_paths(path)
        if len(noisy_paths) == 0:
            print(f'no noisy images found in {path}')
            exit(0)
        for path in tqdm(noisy_paths):
            os.remove(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='path where generate or remove noisy image')
    parser.add_argument('--count', type=int, default=5, help='generate count per image')
    parser.add_argument('--generate', action='store_true', help='generate noisy images')
    parser.add_argument('--remove', action='store_true', help='remove noisy images')
    args = parser.parse_args()
    if not args.generate and not args.remove:
        print('--generate or --remove must be included')
        exit(0)
    if args.generate and args.remove:
        print('--generate and --remove cannot be used at the same time')
        exit(0)
    noisy_data_generator = NoisyDataGenerator(args.count)
    if args.generate:
        noisy_data_generator.generate(args.path)
    elif args.remove:
        noisy_data_generator.remove(args.path)

