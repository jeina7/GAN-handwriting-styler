import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

import os
import time, datetime
import numpy as np

from utils import scale_back, merge, save_concat_images, denorm_image
from dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from ops import conv2d, deconv2d, lrelu, fc, batch_norm
from ops import init_embedding, embedding_lookup, conditional_instance_norm
from models import Encoder, Decoder, Discriminator, Generator

GPU = torch.cuda.is_available()
FONTS_NUM = 30
EMBEDDING_DIM = 128
BATCH_SIZE = 32
IMG_SIZE = 64
DATA_DIR = './data/'

EMBEDDINGS = init_embedding(FONTS_NUM, EMBEDDING_DIM)
if GPU:
    EMBEDDINGS = EMBEDDINGS.cuda()

En = Encoder()
De = Decoder()
D = Discriminator(category_num=FONTS_NUM)
    
# L1 loss, binary real/fake loss, category loss, constant loss
if GPU:
    l1_criterion = nn.L1Loss(size_average=True).cuda()
    bce_criterion = nn.BCEWithLogitsLoss(size_average=True).cuda()
    mse_criterion = nn.MSELoss(size_average=True).cuda()
else:
    l1_criterion = nn.L1Loss(size_average=True)
    bce_criterion = nn.BCEWithLogitsLoss(size_average=True)
    mse_criterion = nn.MSELoss(size_average=True)


# optimizer
G_parameters = list(En.parameters()) + list(De.parameters())
g_optimizer = torch.optim.Adam(G_parameters, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(D.parameters(), betas=(0.5, 0.999))

if GPU:
    En.cuda()
    De.cuda()
    D.cuda()

    
def train(max_epoch, schedule, log_step, sample_step, lr, data_dir, sample_path, fine_tune=False, flip_labels=False):
    if fine_tune:
        L1_penalty, Lconst_penalty = 500, 1000
    else:
        L1_penalty, Lconst_penalty = 100, 15
        
    count = 0
    l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()
    
    data_provider = TrainDataProvider(data_dir, filter_by=range(FONTS_NUM))
    total_batches = data_provider.compute_total_batch_num(BATCH_SIZE)

    # fixed_source
    train_batch_iter = data_provider.get_train_iter(BATCH_SIZE)
    for _, _, batch_images in train_batch_iter:
        fixed_batch = batch_images
        fixed_source = fixed_batch[:, 1, :, :].reshape(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
        if GPU:
            fixed_source = fixed_source.cuda()
        break
    
    for epoch in range(max_epoch):
        if (epoch + 1) % schedule == 0:
            updated_lr = lr / 2
            updated_lr = max(updated_lr, 0.0002)
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = updated_lr
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = updated_lr
            print("decay learning rate from %.5f to %.5f" % (lr, updated_lr))
            lr = updated_lr
            
        for i, batch in enumerate(train_batch_iter):
            count += 1
            labels, codes, batch_images = batch
            embedding_ids = labels
            if GPU:
                batch_images = batch_images.cuda()
            if flip_labels:
                np.random.shuffle(embedding_ids)
                
            # target / source images
            real_target = batch_images[:, 0, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])
            real_source = batch_images[:, 1, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])
            
            # generate fake image form source image
            fake_target, encoded_source = Generator(real_source, En, De, EMBEDDINGS, embedding_ids, GPU=GPU)
            
            real_TS = torch.cat([real_source, real_target], dim=1)
            fake_TS = torch.cat([real_source, fake_target], dim=1)
            
            # Scoring with Discriminator
            real_score, real_score_logit, real_cat_logit = D(real_TS)
            fake_score, fake_score_logit, fake_cat_logit = D(fake_TS)
            
            # Get encoded fake image to calculate constant loss
            encoded_fake = En(fake_target)[0]
            const_loss = Lconst_penalty * mse_criterion(encoded_source, encoded_fake)
            
            # category loss
            real_category = torch.from_numpy(np.eye(FONTS_NUM)[embedding_ids]).float()
            if GPU:
                real_category = real_category.cuda()
            real_category_loss = bce_criterion(real_cat_logit, real_category)
            fake_category_loss = bce_criterion(fake_cat_logit, real_category)
            category_loss = 0.5 * (real_category_loss + fake_category_loss)
            
            # labels
            if GPU:
                one_labels = torch.ones([BATCH_SIZE, 1]).cuda()
                zero_labels = torch.zeros([BATCH_SIZE, 1]).cuda()
            else:
                one_labels = torch.ones([BATCH_SIZE, 1])
                zero_labels = torch.zeros([BATCH_SIZE, 1])
            
            # binary loss - T/F
            real_binary_loss = bce_criterion(real_score_logit, one_labels)
            fake_binary_loss = bce_criterion(fake_score_logit, zero_labels)
            binary_loss = real_binary_loss + fake_binary_loss
            
            # L1 loss between real and fake images
            l1_loss = L1_penalty * l1_criterion(real_target, fake_target)
            
            # cheat loss for generator to fool discriminator
            cheat_loss = bce_criterion(fake_score_logit, one_labels)
            
            # g_loss, d_loss
            g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss
            d_loss = binary_loss + category_loss
            
            # train Discriminator
            D.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            
            # train Generator
            En.zero_grad()
            De.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()            
            
            # loss data
            l1_losses.append(l1_loss.data)
            const_losses.append(const_loss.data)
            category_losses.append(category_loss.data)
            d_losses.append(d_loss.data)
            g_losses.append(g_loss.data)
            
            # logging
            if (i+1) % log_step == 0:
                time_ = time.time()
                time_stamp = datetime.datetime.fromtimestamp(time_).strftime('%H:%M:%S')
                log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
                             (epoch+1, max_epoch, i+1, total_batches, l1_loss.item(), d_loss.item(), g_loss.item())
                print(time_stamp, log_format)
                
            # save image
            if (i+1) % sample_step == 0:
                fixed_fake_images = Generator(fixed_source, En, De, EMBEDDINGS, embedding_ids, GPU=GPU)[0]
                save_image(denorm_image(fixed_fake_images.data), \
                           os.path.join(sample_path, 'fake_samples-%d-%d.png' % (epoch+1, i+1)), nrow=8)

    return l1_losses, const_losses, category_losses, d_losses, g_losses
