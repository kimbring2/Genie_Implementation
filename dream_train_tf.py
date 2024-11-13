import glob
import os
import math
import numpy as np
import random
import pathlib
from tqdm import tqdm
import time
import cv2
from pathlib import Path
from collections import defaultdict
from functools import partial
import sys
import math

import tensorflow as tf
import tensorflow_probability as tfp

import utils
import networks

tfd = tfp.distributions

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

num_actions = 7
num_hidden_units = 2048
lam_model = networks.InverseActionPolicy(num_actions, num_hidden_units)
lam_model.load_weights("model/IDM_Model_80")

latent_dim = 512
num_embeddings = 512
tokenizer_model = networks.VQ_VAE(latent_dim, num_embeddings)
tokenizer_model.load_weights("model/CoinRun_VAVAQ_Model_60.ckpt")

num_layers = 10
vocab_size = 512
num_actions = 7
max_tokens = 340
embed_dim = 256
max_blocks = 20
sequence_length = max_blocks
tokens_per_block = 17
all_but_last_obs_tokens_pattern = np.ones(tokens_per_block)
all_but_last_obs_tokens_pattern[-2] = 0
act_tokens_pattern = np.zeros(tokens_per_block)
act_tokens_pattern[-1] = 1
obs_tokens_pattern = 1 - act_tokens_pattern

obs_slicer = networks.Slicer(max_blocks, obs_tokens_pattern)
act_slicer = networks.Slicer(max_blocks, act_tokens_pattern)
head_obs_slicer = networks.Slicer(max_blocks, all_but_last_obs_tokens_pattern)

world_model = networks.WorldModel(max_blocks=max_blocks, tokens_per_block=tokens_per_block, max_tokens=max_tokens, obs_vocab_size=vocab_size, 
                                  act_vocab_size=num_actions, units=embed_dim, dropout_rate=0.1, num_layers=num_layers, num_heads=4, 
                                  obs_slicer=obs_slicer, act_slicer=act_slicer, head_obs_slicer=head_obs_slicer)


optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.001)

class CoinRunDataset:
    def __init__(self, data_dir, set='train', batch_size=32):
        self.data_dir = data_dir
        self.set = 'train'
        self.batch_size = batch_size
        self.filenames = os.listdir(data_dir)
        self.max_frames = 20
        self.a_width = 1
        
        self.data_length = len(self.filenames)
        print("Length of %s data: " % set, self.data_length)

    def __len__(self):
        return self.data_length

    def __iter__(self):
        for _ in range(0, self.data_length):
            file_idx = random.randint(0, self.data_length - 1)
            fname = self.filenames[file_idx]
            
            if not fname.endswith('npz'): 
                continue
    
            file_path = os.path.join(self.data_dir, fname)
            data = np.load(file_path)
            img = data['obs']

            video_length = img.shape[0]
            
            action = np.reshape(data['action'], newshape=[-1, self.a_width])
            reward = data['reward']
            done = data['done']
            N = data['N']

            for _ in range(0, 10):
                offset = random.randint(0, video_length - self.max_frames - 1)
                
                img_sub = img[offset:offset + self.max_frames]
                action_sub = action[offset:offset + self.max_frames]
                reward_sub = reward[offset:offset + self.max_frames]
                done_sub = done[offset:offset + self.max_frames]
                N_sub = N[offset:offset + self.max_frames]

                yield img_sub, action_sub, reward_sub, done_sub, N_sub

    __call__ = __iter__


batch_size = 16
data_dir = "record_single"
ds_gen = CoinRunDataset(data_dir=data_dir, set='train', batch_size=batch_size)
ds = tf.data.Dataset.from_generator(ds_gen, (tf.float32, tf.float32, tf.float32, tf.bool, tf.int16))
ds = ds.batch(batch_size, drop_remainder=True)


for epoch in range(0, 1000000):
    #print("epoch: ", epoch)
    train_losses = []
    for step, batch in enumerate(ds):
        #print("step: ", step)
        video, action_, reward, done, N = batch
        #video, _, reward, done, N = batch
        #print("video.shape: ", video.shape)
        #print("action.shape 1: ", action_.shape)

        act_pi = lam_model(video, training=False)
        act_dist = tfd.Categorical(logits=act_pi)
        action = act_dist.sample()
        action = tf.cast(action, tf.float32)
        action = tf.expand_dims(action, 2)
        #print("action.shape 2: ", action.shape)
        #print("    action: ", tf.squeeze(action))
        #print("pre_action.tolist(): ", pre_action.tolist())
        #print("pre_action: ", pre_action)
        #print("")

        #for frame in video[0]:
            #print("frame.shape: ", frame.shape)
            #print("frame.numpy(): ", frame.numpy())
            #cv2.imshow('Frame', frame.numpy())
            #cv2.waitKey(1)

        # video.shape:  (16, 20, 64, 64, 3)
        # action.shape:  (16, 20, 1)
        # reward.shape:  (16, 20)
        # done.shape:  (16, 20)

        batch_obs = video - 0.5
        batch_act = action
        batch_rew = reward
        batch_end = done
        #batch_mask_padding = batch['mask_padding'].cpu().detach().numpy()
    
        #batch_obs = np.transpose(batch_obs, [0, 1, 3, 4, 2])
        batch_obs = np.reshape(batch_obs, [batch_obs.shape[0] * batch_obs.shape[1], batch_obs.shape[2], batch_obs.shape[3], batch_obs.shape[4]])
        
        encoder_outputs = tokenizer_model.encoder(batch_obs)
        #print("encoder_outputs.shape: ", encoder_outputs.shape)
        quantized_latents, encoding_indices = tokenizer_model.vq_layer(encoder_outputs)
        
        #print("encoding_indices.shape: ", encoding_indices.shape)
        obs_tokens = tf.reshape(encoding_indices, [batch_obs.shape[0], tokens_per_block - 1])
        #print("obs_tokens.shape 1: ", obs_tokens.shape)

        obs_tokens = tf.reshape(encoding_indices, [batch_size, sequence_length, -1])
        obs_tokens = tf.cast(obs_tokens, tf.int32)
        
        #act_tokens = tf.expand_dims(batch_act, -1)
        act_tokens = batch_act
        act_tokens = tf.cast(act_tokens, tf.int32)

        #print("obs_tokens.shape: ", obs_tokens.shape)
        #print("act_tokens.shape: ", act_tokens.shape)
        tokens = tf.concat([obs_tokens, act_tokens], 2)
        tokens = tf.reshape(tokens, [batch_size, -1])
        tokens = tf.cast(tokens, tf.float32)
        
        batch_size = tokens.shape[0]

        with tf.GradientTape() as tape:
            logits_obs, logits_rew, logits_end, _, _ = world_model(tokens, decode_step=None, k_cache_array=None, v_cache_array=None)
            logits_obs = logits_obs[:, :-1]
            logits_obs = tf.reshape(logits_obs, [batch_size * logits_obs.shape[1], -1]) 
            logits_end = tf.reshape(logits_end, [batch_size * logits_end.shape[1], -1])

            labels_obs, labels_rew, labels_end = utils.compute_labels_world_model(obs_tokens, batch_rew, batch_end)
            
            loss_obs = tf.keras.losses.sparse_categorical_crossentropy(labels_obs, logits_obs, from_logits=True, ignore_class=-100)
            count_nonzero = tf.cast(tf.math.count_nonzero(loss_obs), tf.float32)
            loss_obs = tf.reduce_sum(loss_obs) / count_nonzero
            
            loss_end = tf.keras.losses.sparse_categorical_crossentropy(labels_end, logits_end, from_logits=True, ignore_class=-100)
            count_nonzero = tf.cast(tf.math.count_nonzero(loss_end), tf.float32)
            loss_end = tf.reduce_sum(loss_end) / count_nonzero

            loss = loss_obs + loss_end
            
            train_losses.append(loss)
            print("step: {}, loss: {}".format(step, loss))

        grads = tape.gradient(loss, world_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, world_model.trainable_variables))

        #break

    mean_loss_train = np.mean(train_losses)
    print("Epoch: {}, Mean train loss: {}".format(epoch, mean_loss_train))
    world_model.save_weights('model/world_model_' + str(epoch))

    print("")