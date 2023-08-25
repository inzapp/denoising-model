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
import argparse

from denoising_model import TrainingConfig, DenoisingModel


if __name__ == '__main__':
    config = TrainingConfig(
        train_image_path='/train_data/coco/train',
        validation_image_path='/train_data/coco/validation',
        input_rows=256,
        input_cols=256,
        input_type='gray',  # available types : [gray, rgb, nv12, nv21]
        lr=0.0003,
        warm_up=0.1,
        batch_size=2,
        max_noise=30,
        iterations=100000,
        save_interval=10000,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--type', type=str, default='', help='pretrained model input type : [gray, rgb, nv12, nv21]')
    parser.add_argument('--predict', action='store_true', help='prediction using given dataset')
    parser.add_argument('--evaluate', action='store_true', help='evaluate using given dataset')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='image or video path for prediction or evaluation')
    parser.add_argument('--r', action='store_true', help='find images recursively')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    args = parser.parse_args()
    if args.model != '':
        config.pretrained_model_path = args.model
    if args.type != '':
        config.input_type = args.type
    denoising_model = DenoisingModel(config=config, training=not (args.predict or args.evaluate))
    if args.predict:
        denoising_model.predict_images(image_path=args.path, dataset=args.dataset, save_count=args.save_count, recursive=args.r)
    elif args.evaluate:
        denoising_model.evaluate(image_path=args.path, dataset=args.dataset, recursive=args.r)
    else:
        denoising_model.train()

