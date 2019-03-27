#!/usr/bin/env python
# -*- coding:utf-8

import argparse
import os

import numpy as np

import data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('x', help='.mat file')
    parser.add_argument('-o', '--out_dir', default=os.getcwd(), help='Directory to save images in')
    return parser.parse_args()


def main():
    args = parse_args()

    x = np.load(args.x) if args.x.endswith('.npy') else np.squeeze(data.load_data(args.x))

    imdir = args.out_dir
    if not os.path.isdir(imdir):
        os.makedirs(imdir)

    for i, im in enumerate(x):
        path = os.path.join(imdir, '{:04d}.png'.format(i))
        data.save_img(im, path)


if __name__ == '__main__':
    main()
