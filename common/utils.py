# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
from io import BytesIO

import matplotlib.pyplot as plt


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def denorm_image(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h = img.shape
    enlarged = misc.imresize(img, [nw, nh])
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def save_gif(gif_path, image_path, file_name):
    filenames = sorted(glob.glob(os.path.join(image_path, "*.png")))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(gif_path, file_name), images)


def show_comparison(font_num, real_targets, fake_targets, show_num=8):
    plt.figure(figsize=(14, show_num//2+1))
    for idx in range(show_num):
        plt.subplot(show_num//4, 8, 2*idx+1)
        plt.imshow(real_targets[font_num][idx].reshape(128, 128), cmap='gray')
        plt.title("Real [%d]" % font_num)
        plt.axis('off')

        plt.subplot(show_num//4, 8, 2*idx+2)
        plt.imshow(fake_targets[font_num][idx].reshape(128, 128), cmap='gray')
        plt.title("Fake [%d]" % font_num)
        plt.axis('off')
    plt.show()
