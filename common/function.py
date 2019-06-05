# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


def batch_norm(c_out, momentum=0.1):
    return nn.BatchNorm2d(c_out, momentum=momentum)


def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, dilation=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def deconv2d(c_in, c_out, k_size=3, stride=1, pad=1, dilation=1, bn=True, dropout=False, p=0.5):
    layers = []
    layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if dropout:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


def lrelu(leak=0.2):
    return nn.LeakyReLU(leak)


def dropout(p=0.2):
    return nn.Dropout(p)


def fc(input_size, output_size):
    return nn.Linear(input_size, output_size)
    
    
def init_embedding(embedding_num, embedding_dim, stddev=0.01):
    embedding = torch.randn(embedding_num, embedding_dim) * stddev
    embedding = embedding.reshape((embedding_num, 1, 1, embedding_dim))
    return embedding


def embedding_lookup(embeddings, embedding_ids, GPU=False):
    batch_size = len(embedding_ids)
    embedding_dim = embeddings.shape[3]
    local_embeddings = []
    for id_ in embedding_ids:
        if GPU:
            local_embeddings.append(embeddings[id_].cpu().numpy())
        else:
            local_embeddings.append(embeddings[id_].data.numpy())
    local_embeddings = torch.from_numpy(np.array(local_embeddings))
    if GPU:
        local_embeddings = local_embeddings.cuda()
    local_embeddings = local_embeddings.reshape(batch_size, embedding_dim, 1, 1)
    return local_embeddings


def interpolated_embedding_lookup(embeddings, interpolated_embedding_ids, grid):
    batch_size = len(interpolated_embedding_ids)
    interpolated_embeddings = []
    embedding_dim = embeddings.shape[3]

    for id_ in interpolated_embedding_ids:
        interpolated_embeddings.append((embeddings[id_[0]] * (1 - grid) + embeddings[id_[1]] * grid).cpu().numpy())
    interpolated_embeddings = torch.from_numpy(np.array(interpolated_embeddings)).cuda()
    interpolated_embeddings = interpolated_embeddings.reshape(batch_size, embedding_dim, 1, 1)
    return interpolated_embeddings


def Interpolate_encoded_sources(interpolated_embedding_ids, grid, all_charid, font_id):
    interpolated_encoded_sources = []

    for id_ in interpolated_embedding_ids:
        index_0 = all_charid[font_id]['font'].index(id_[0])
        index_1 = all_charid[font_id]['font'].index(id_[1])
        interpolated_encoded_source = (all_charid[font_id]['encoded_source'][index_0] * (1 - grid) + \
                                       all_charid[font_id]['encoded_source'][index_1]).cpu().detach().numpy()
        interpolated_encoded_sources.append(interpolated_encoded_source)
    interpolated_encoded_sources = torch.from_numpy(np.array(interpolated_encoded_sources)).cuda()
    return interpolated_encoded_sources


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]
        scale = tf.get_variable("scale", [labels_num, output_filters], tf.float32, tf.constant_initializer(1.0))
        shift = tf.get_variable("shift", [labels_num, output_filters], tf.float32, tf.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x, [1, 2], keep_dims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        batch_scale = tf.reshape(tf.nn.embedding_lookup([scale], ids=ids), [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup([shift], ids=ids), [batch_size, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z
