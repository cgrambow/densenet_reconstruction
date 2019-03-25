#!/usr/bin/env python
# -*- coding:utf-8

import argparse
import os

import numpy as np

import data
import nn


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-x', '--x_train', dest='x_path', nargs='+', help='Path to noisy measurements')
    parser.add_argument('-y', '--y_train', dest='y_path', nargs='+', help='Path to ground truth')
    parser.add_argument('-s', '--save_dir', default=os.getcwd(), help='Directory to save weights in')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('-v', '--validation_split', type=float, default=0.05,
                        help='Fraction of data to use for validation')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-xt', '--x_test', dest='x_test_path', help='Path to noisy measurements (test data)')
    parser.add_argument('-yt', '--y_test', dest='y_test_path', help='Path to ground truth(test data)')
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.x_path) > 1:
        x = data.load_multiple(args.x_path)
    else:
        x = data.load_data(args.x_path[0])
    if len(args.y_path) > 1:
        y = data.load_multiple(args.y_path)
    else:
        y = data.load_data(args.y_path[0])
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    x /= x.max()  # Scale images
    y /= y.max()

    model = nn.build()
    nn.train(model, x, y, args.save_dir,
             batch_size=args.batch_size,
             max_epochs=args.epochs,
             validation_split=args.validation_split,
             patience=args.patience)

    if args.x_test_path is not None and args.y_test_path is not None:
        x_test = data.load_data(args.x_test_path)
        y_test = data.load_data(args.y_test_path)
        assert len(x_test) == len(y_test)
        x_test /= x_test.max()
        y_test /= y_test.max()
        test_loss = model.evaluate(x_test, y_test)
        print('Test loss:', test_loss)


if __name__ == '__main__':
    main()
