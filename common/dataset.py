# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle as pickle
import numpy as np
import random
import os
import torch
from .utils import pad_seq, bytes_to_file, read_split_image, shift_and_resize_image, normalize_image


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A)
            img_A = img_A.reshape(1, len(img_A), len(img_A[0]))
            img_B = normalize_image(img_B)
            img_B = img_B.reshape(1, len(img_B), len(img_B[0]))
            return np.concatenate([img_A, img_B], axis=0)
        finally:
            img.close()
            
    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            image = [process(e[1]) for e in batch]
            image = np.array(image).astype(np.float32)
            image = torch.from_numpy(image)
            # stack into tensor
            yield labels, image

    return batch_iter()


class PickledImageProvider(object):
    def __init__(self, obj_path, verbose):
        self.obj_path = obj_path
        self.verbose = verbose
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                except EOFError:
                    break
                except Exception:
                    pass
            if self.verbose:
                print("unpickled total %d examples" % len(examples))
            return examples


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None, verbose=True):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path, verbose)
        self.val = PickledImageProvider(self.val_path, verbose)
        if self.filter_by:
            if verbose:
                print("filter by label ->", filter_by)
            self.train.examples = [e for e in self.train.examples if e[0] in self.filter_by]
            self.val.examples = [e for e in self.val.examples if e[0] in self.filter_by]
        if verbose:
            print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=True)

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
            for labels, examples in val_batch_iter:
                yield labels, examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path
    
    