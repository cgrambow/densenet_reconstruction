#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import tables


def load_data(path):
    name, _ = os.path.splitext(os.path.basename(path))
    dataf = tables.open_file(path)
    name2 = name.rstrip(string.digits)
    if name2 != name and name2.endswith('_'):
        name = name2[:-1]  # Remove trailing underscore
    data = dataf.get_node('/' + name)[:]
    dataf.close()
    data = np.swapaxes(np.squeeze(data), 1, 2)
    return np.expand_dims(data, axis=-1).astype(np.float32)


def load_multiple(paths):
    assert len(paths) > 1
    data = load_data(paths[0])
    for path in paths[1:]:
        data = np.concatenate((data, load_data(path)))
    return data


def save_img(im, path):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(im, aspect='equal')
    ax.set_axis_off()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=extent, pad_inches=0)
