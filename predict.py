#!/usr/bin/env python
# -*- coding:utf-8

import argparse
import os

import keras
import numpy as np

import data
import nn


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('x', help='Path to noisy measurements')
    parser.add_argument('model', help='Path to trained model')
    parser.add_argument('-y', '--y', help='Path to ground truth')
    parser.add_argument('-m', '--xymax', help='Path to image scales')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(), help='Directory to save predictions in')
    parser.add_argument('--save_imgs', action='store_true', help='Save individual images')
    return parser.parse_args()


def main():
    args = parse_args()

    x = data.load_data(args.x)
    model = keras.models.load_model(args.model, custom_objects={'npcc': nn.npcc})

    xmax, ymax = np.load(args.xymax) if args.xymax is not None else (x.max(), 1.0)
    x /= xmax

    if args.y is not None:
        y = data.load_data(args.y)
        ymax = y.max() if args.xymax is None else ymax
        y /= ymax
        pcc = - model.evaluate(x, y, verbose=0)
        print('PCC:', pcc)

    ypred = np.squeeze(model.predict(x, verbose=0))
    ypred *= ymax

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    np.save(os.path.join(args.out_dir, 'predictions.npy'), ypred)

    if args.save_imgs:
        imdir = os.path.join(args.out_dir, 'imgs')
        if not os.path.isdir(imdir):
            os.makedirs(imdir)

        for i, im in enumerate(ypred):
            path = os.path.join(imdir, '{:04d}.png'.format(i))
            data.save_img(im, path)


if __name__ == '__main__':
    main()
