import os, glob, time, datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image

from common.dataset import TrainDataProvider
from common.function import init_embedding
from common.models import Encoder, Decoder, Discriminator, Generator
from common.utils import denorm_image, centering_image


class Trainer:
    
    def __init__(self, GPU, data_dir, fixed_dir, fonts_num, batch_size, img_size):
        self.GPU = GPU
        self.data_dir = data_dir
        self.fixed_dir = fixed_dir
        self.fonts_num = fonts_num
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.embeddings = torch.load(os.path.join(fixed_dir, 'EMBEDDINGS.pkl'))
        self.embedding_num = self.embeddings.shape[0]
        self.embedding_dim = self.embeddings.shape[3]
        
        self.fixed_source = torch.load(os.path.join(fixed_dir, 'fixed_source.pkl'))
        self.fixed_target = torch.load(os.path.join(fixed_dir, 'fixed_target.pkl'))
        self.fixed_label = torch.load(os.path.join(fixed_dir, 'fixed_label.pkl'))
        
        self.data_provider = TrainDataProvider(self.data_dir)
        self.total_batches = self.data_provider.compute_total_batch_num(self.batch_size)
        print("total batches:", self.total_batches)


    def train(self, max_epoch, schedule, save_path, to_model_path, lr=0.001, \
              log_step=100, sample_step=350, fine_tune=False, flip_labels=False, \
              restore=None, from_model_path=False, with_charid=False, \
              freeze_encoder=False, save_nrow=8, model_save_step=None, resize_fix=90):

        # Fine Tuning coefficient
        if not fine_tune:
            L1_penalty, Lconst_penalty = 100, 15
        else:
            L1_penalty, Lconst_penalty = 500, 1000

        # Get Models
        En = Encoder()
        De = Decoder()
        D = Discriminator(category_num=self.fonts_num)
        if self.GPU:
            En.cuda()
            De.cuda()
            D.cuda()

        # Use pre-trained Model
        # restore에 [encoder_path, decoder_path, discriminator_path] 형태로 인자 넣기
        if restore:
            encoder_path, decoder_path, discriminator_path = restore
            prev_epoch = int(encoder_path.split('-')[0])
            En.load_state_dict(torch.load(os.path.join(from_model_path, encoder_path)))
            De.load_state_dict(torch.load(os.path.join(from_model_path, decoder_path)))
            D.load_state_dict(torch.load(os.path.join(from_model_path, discriminator_path)))
            print("%d epoch trained model has restored" % prev_epoch)
        else:
            prev_epoch = 0
            print("New model training start")


        # L1 loss, binary real/fake loss, category loss, constant loss
        if self.GPU:
            l1_criterion = nn.L1Loss(size_average=True).cuda()
            bce_criterion = nn.BCEWithLogitsLoss(size_average=True).cuda()
            mse_criterion = nn.MSELoss(size_average=True).cuda()
        else:
            l1_criterion = nn.L1Loss(size_average=True)
            bce_criterion = nn.BCEWithLogitsLoss(size_average=True)
            mse_criterion = nn.MSELoss(size_average=True)


        # optimizer
        if freeze_encoder:
            G_parameters = list(De.parameters())
        else:
            G_parameters = list(En.parameters()) + list(De.parameters())
        g_optimizer = torch.optim.Adam(G_parameters, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(D.parameters(), betas=(0.5, 0.999))

        # losses lists
        l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()

        # training
        count = 0
        for epoch in range(max_epoch):
            if (epoch + 1) % schedule == 0:
                updated_lr = max(lr/2, 0.0002)
                for param_group in d_optimizer.param_groups:
                    param_group['lr'] = updated_lr
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = updated_lr
                if lr !=  updated_lr:
                    print("decay learning rate from %.5f to %.5f" % (lr, updated_lr))
                lr = updated_lr

            train_batch_iter = self.data_provider.get_train_iter(self.batch_size, \
                                                            with_charid=with_charid)   
            for i, batch in enumerate(train_batch_iter):
                if with_charid:
                    font_ids, char_ids, batch_images = batch
                else:
                    font_ids, batch_images = batch
                embedding_ids = font_ids
                if self.GPU:
                    batch_images = batch_images.cuda()
                if flip_labels:
                    np.random.shuffle(embedding_ids)

                # target / source images
                real_target = batch_images[:, 0, :, :]
                real_target = real_target.view([self.batch_size, 1, self.img_size, self.img_size])
                real_source = batch_images[:, 1, :, :]
                real_source = real_source.view([self.batch_size, 1, self.img_size, self.img_size])
                
                # centering
                for idx, (image_S, image_T) in enumerate(zip(real_source, real_target)):
                    image_S = image_S.cpu().detach().numpy().reshape(self.img_size, self.img_size)
                    image_S = centering_image(image_S, resize_fix=90)
                    real_source[idx] = torch.tensor(image_S).view([1, self.img_size, self.img_size])
                    image_T = image_T.cpu().detach().numpy().reshape(self.img_size, self.img_size)
                    image_T = centering_image(image_T, resize_fix=resize_fix)
                    real_target[idx] = torch.tensor(image_T).view([1, self.img_size, self.img_size])

                # generate fake image form source image
                fake_target, encoded_source, _ = Generator(real_source, En, De, \
                                                           self.embeddings, embedding_ids, \
                                                           GPU=self.GPU, encode_layers=True)

                real_TS = torch.cat([real_source, real_target], dim=1)
                fake_TS = torch.cat([real_source, fake_target], dim=1)

                # Scoring with Discriminator
                real_score, real_score_logit, real_cat_logit = D(real_TS)
                fake_score, fake_score_logit, fake_cat_logit = D(fake_TS)

                # Get encoded fake image to calculate constant loss
                encoded_fake = En(fake_target)[0]
                const_loss = Lconst_penalty * mse_criterion(encoded_source, encoded_fake)

                # category loss
                real_category = torch.from_numpy(np.eye(self.fonts_num)[embedding_ids]).float()
                if self.GPU:
                    real_category = real_category.cuda()
                real_category_loss = bce_criterion(real_cat_logit, real_category)
                fake_category_loss = bce_criterion(fake_cat_logit, real_category)
                category_loss = 0.5 * (real_category_loss + fake_category_loss)

                # labels
                if self.GPU:
                    one_labels = torch.ones([self.batch_size, 1]).cuda()
                    zero_labels = torch.zeros([self.batch_size, 1]).cuda()
                else:
                    one_labels = torch.ones([self.batch_size, 1])
                    zero_labels = torch.zeros([self.batch_size, 1])

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
                l1_losses.append(int(l1_loss.data))
                const_losses.append(int(const_loss.data))
                category_losses.append(int(category_loss.data))
                d_losses.append(int(d_loss.data))
                g_losses.append(int(g_loss.data))

                # logging
                if (i+1) % log_step == 0:
                    time_ = time.time()
                    time_stamp = datetime.datetime.fromtimestamp(time_).strftime('%H:%M:%S')
                    log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
                                 (int(prev_epoch)+epoch+1, int(prev_epoch)+max_epoch, \
                                  i+1, self.total_batches, l1_loss.item(), d_loss.item(), g_loss.item())
                    print(time_stamp, log_format)

                # save image
                if (i+1) % sample_step == 0:
                    fixed_fake_images = Generator(self.fixed_source, En, De, \
                                                  self.embeddings, self.fixed_label, GPU=self.GPU)[0]
                    save_image(denorm_image(fixed_fake_images.data), \
                               os.path.join(save_path, 'fake_samples-%d-%d.png' % \
                                            (int(prev_epoch)+epoch+1, i+1)), \
                               nrow=save_nrow, pad_value=255)

            if not model_save_step:
                model_save_step = 5
            if (epoch+1) % model_save_step == 0:
                now = datetime.datetime.now()
                now_date = now.strftime("%m%d")
                now_time = now.strftime('%H:%M')
                torch.save(En.state_dict(), os.path.join(to_model_path, \
                                                         '%d-%s-%s-Encoder.pkl' % \
                                                         (int(prev_epoch)+epoch+1, \
                                                          now_date, now_time)))
                torch.save(De.state_dict(), os.path.join(to_model_path, \
                                                         '%d-%s-%s-Decoder.pkl' % \
                                                         (int(prev_epoch)+epoch+1, \
                                                          now_date, now_time)))
                torch.save(D.state_dict(), os.path.join(to_model_path, \
                                                        '%d-%s-%s-Discriminator.pkl' % \
                                                        (int(prev_epoch)+epoch+1, \
                                                         now_date, now_time)))

        # save model
        total_epoch = int(prev_epoch) + int(max_epoch)
        end = datetime.datetime.now()
        end_date = end.strftime("%m%d")
        end_time = end.strftime('%H:%M')
        torch.save(En.state_dict(), os.path.join(to_model_path, \
                                                 '%d-%s-%s-Encoder.pkl' % \
                                                 (total_epoch, end_date, end_time)))
        torch.save(De.state_dict(), os.path.join(to_model_path, \
                                                 '%d-%s-%s-Decoder.pkl' % \
                                                 (total_epoch, end_date, end_time)))
        torch.save(D.state_dict(), os.path.join(to_model_path, \
                                                '%d-%s-%s-Discriminator.pkl' % \
                                                (total_epoch, end_date, end_time)))
        losses = [l1_losses, const_losses, category_losses, d_losses, g_losses]
        torch.save(losses, os.path.join(to_model_path, '%d-losses.pkl' % total_epoch))

        return l1_losses, const_losses, category_losses, d_losses, g_losses


def interpolation(data_provider, grids, fixed_char_ids, interpolated_font_ids, embeddings, \
                  En, De, batch_size, img_size=128, save_nrow=6, save_path=False, GPU=True):
    
    train_batch_iter = data_provider.get_train_iter(batch_size, with_charid=True)
    
    for grid_idx, grid in enumerate(grids):
        train_batch_iter = data_provider.get_train_iter(batch_size, with_charid=True)
        grid_results = {from_to: {charid: None for charid in fixed_char_ids} \
                        for from_to in interpolated_font_ids}

        for i, batch in enumerate(train_batch_iter):
            font_ids_from, char_ids, batch_images = batch
            font_filter = [i[0] for i in interpolated_font_ids]
            font_filter_plus = font_filter + [font_filter[0]]
            font_ids_to = [font_filter_plus[font_filter.index(i)+1] for i in font_ids_from]
            batch_images = batch_images.cuda()

            real_sources = batch_images[:, 1, :, :].view([batch_size, 1, img_size, img_size])
            real_targets = batch_images[:, 0, :, :].view([batch_size, 1, img_size, img_size])

            for idx, (image_S, image_T) in enumerate(zip(real_sources, real_targets)):
                image_S = image_S.cpu().detach().numpy().reshape(img_size, img_size)
                image_S = centering_image(image_S, resize_fix=100)
                real_sources[idx] = torch.tensor(image_S).view([1, img_size, img_size])
                image_T = image_T.cpu().detach().numpy().reshape(img_size, img_size)
                image_T = centering_image(image_T, resize_fix=100)
                real_targets[idx] = torch.tensor(image_T).view([1, img_size, img_size])
                
            encoded_source, encode_layers = En(real_sources)

            interpolated_embeddings = []
            embedding_dim = embeddings.shape[3]
            for from_, to_ in zip(font_ids_from, font_ids_to):
                interpolated_embeddings.append((embeddings[from_] * (1 - grid) + \
                                                embeddings[to_] * grid).cpu().numpy())
            interpolated_embeddings = torch.tensor(interpolated_embeddings).cuda()
            interpolated_embeddings = interpolated_embeddings.reshape(batch_size, embedding_dim, 1, 1)

            # generate fake image with embedded source
            interpolated_embedded = torch.cat((encoded_source, interpolated_embeddings), 1)
            fake_targets = De(interpolated_embedded, encode_layers)

            # [(0)real_S, (1)real_T, (2)fake_T]
            for fontid, charid, real_S, real_T, fake_T in zip(font_ids_from, char_ids, \
                                                              real_sources, real_targets, \
                                                              fake_targets):
                font_from = fontid
                font_to = font_filter_plus[font_filter.index(fontid)+1]
                from_to = (font_from, font_to)
                grid_results[from_to][charid] = [real_S, real_T, fake_T]

        if save_path:
            for from_to in grid_results.keys():
                image = [grid_results[from_to][charid][2].cpu().detach().numpy() for \
                         charid in fixed_char_ids]
                image = torch.tensor(np.array(image))

                # path
                font_from = str(from_to[0])
                font_to = str(from_to[1])
                grid_idx = str(grid_idx)
                if len(font_from) == 1:
                    font_from = '0' + font_from
                if len(font_to) == 1:
                    font_to = '0' + font_to
                if len(grid_idx) == 1:
                    grid_idx = '0' + grid_idx
                idx = str(interpolated_font_ids.index(from_to))
                if len(idx) == 1:
                    idx = '0' + idx
                file_path = '%s_from_%s_to_%s_grid_%s.png' % (idx, font_from, font_to, grid_idx)

                # save
                save_image(denorm_image(image.data), \
                           os.path.join(save_path, file_path), \
                           nrow=save_nrow, pad_value=255)
    
    return grid_results