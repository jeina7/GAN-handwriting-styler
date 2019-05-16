# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import pickle as pickle
import random


def pickle_examples(from_dir, train_path, val_path, train_val_split=0.2):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    paths = glob.glob(os.path.join(from_dir, "*.png"))
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            print('all data num:', len(paths))
            c = 1
            val_count = 0
            train_count = 0
            for p in paths:
                c += 1
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    example = (label, img_bytes)
                    r = random.random()
                    if r < train_val_split:
                        pickle.dump(example, fv)
                        val_count += 1
                        if val_count % 10000 == 0:
                            print("%d imgs saved in val.obj" % val_count)
                    else:
                        pickle.dump(example, ft)
                        train_count += 1
                        if train_count % 10000 == 0:
                            print("%d imgs saved in train.obj" % train_count)
            print("%d imgs saved in val.obj, end" % val_count)
            print("%d imgs saved in train.obj, end" % train_count)
            return
        
        