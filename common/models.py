# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .function import conv2d, deconv2d, lrelu, fc, embedding_lookup
import warnings
warnings.filterwarnings("ignore")


def Generator(images, En, De, embeddings, embedding_ids, GPU=False, encode_layers=False):
    encoded_source, encode_layers = En(images)
    local_embeddings = embedding_lookup(embeddings, embedding_ids, GPU=GPU)
    if GPU:
        encoded_source = encoded_source.cuda()
        local_embeddings = local_embeddings.cuda()
    embedded = torch.cat((encoded_source, local_embeddings), 1)
    fake_target = De(embedded, encode_layers)
    if encode_layers:
        return fake_target, encoded_source, encode_layers
    else:
        return fake_target, encoded_source


class Encoder(nn.Module):
    
    def __init__(self, img_dim=1, conv_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=2, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=2, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
        self.conv6 = conv2d(conv_dim*8, conv_dim*8)
        self.conv7 = conv2d(conv_dim*8, conv_dim*8)
        self.conv8 = conv2d(conv_dim*8, conv_dim*8)
    
    def forward(self, images):
        encode_layers = dict()
        
        e1 = self.conv1(images)
        encode_layers['e1'] = e1
        e2 = self.conv2(e1)
        encode_layers['e2'] = e2
        e3 = self.conv3(e2)
        encode_layers['e3'] = e3
        e4 = self.conv4(e3)
        encode_layers['e4'] = e4
        e5 = self.conv5(e4)
        encode_layers['e5'] = e5
        e6 = self.conv6(e5)
        encode_layers['e6'] = e6
        e7 = self.conv7(e6)
        encode_layers['e7'] = e7
        encoded_source = self.conv8(e7)
        encode_layers['e8'] = encoded_source
        
        return encoded_source, encode_layers
    
    
class Decoder(nn.Module):
    
    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
        super(Decoder, self).__init__()
        self.deconv1 = deconv2d(embedded_dim, conv_dim*8, dropout=True)
        self.deconv2 = deconv2d(conv_dim*16, conv_dim*8, dropout=True, k_size=4)
        self.deconv3 = deconv2d(conv_dim*16, conv_dim*8, k_size=5, dilation=2, dropout=True)
        self.deconv4 = deconv2d(conv_dim*16, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv5 = deconv2d(conv_dim*16, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv6 = deconv2d(conv_dim*8, conv_dim*2, k_size=4, dilation=2, stride=2)
        self.deconv7 = deconv2d(conv_dim*4, conv_dim*1, k_size=4, dilation=2, stride=2)
        self.deconv8 = deconv2d(conv_dim*2, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    
    def forward(self, embedded, encode_layers):
        
        d1 = self.deconv1(embedded)
        d1 = torch.cat((d1, encode_layers['e7']), dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat((d2, encode_layers['e6']), dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat((d3, encode_layers['e5']), dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat((d4, encode_layers['e4']), dim=1)
        d5 = self.deconv5(d4)
        d5 = torch.cat((d5, encode_layers['e3']), dim=1)
        d6 = self.deconv6(d5)
        d6 = torch.cat((d6, encode_layers['e2']), dim=1)
        d7 = self.deconv7(d6)
        d7 = torch.cat((d7, encode_layers['e1']), dim=1)
        d8 = self.deconv8(d7)        
        fake_target = torch.tanh(d8)
        
        return fake_target
    
    
class Discriminator(nn.Module):
    def __init__(self, category_num, img_dim=2, disc_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv2d(img_dim, disc_dim, bn=False)
        self.conv2 = conv2d(disc_dim, disc_dim*2)
        self.conv3 = conv2d(disc_dim*2, disc_dim*4)
        self.conv4 = conv2d(disc_dim*4, disc_dim*8)
        self.fc1 = fc(disc_dim*8*8*8, 1)
        self.fc2 = fc(disc_dim*8*8*8, category_num)
        
    def forward(self, images):
        batch_size = images.shape[0]
        h1 = self.conv1(images)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        
        tf_loss_logit = self.fc1(h4.reshape(batch_size, -1))
        tf_loss = torch.sigmoid(tf_loss_logit)
        cat_loss = self.fc2(h4.reshape(batch_size, -1))
        
        return tf_loss, tf_loss_logit, cat_loss
    
    