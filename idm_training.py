import numpy as np
import os
import cv2
import random
import time
from typing import Any, List, Sequence, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

import utils
import networks

tfd = tfp.distributions

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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


data_dir = "record_single"
ds_gen = CoinRunDataset(data_dir=data_dir, set='train', batch_size=32)
ds = tf.data.Dataset.from_generator(ds_gen, (tf.float32, tf.float32, tf.float32, tf.bool, tf.int16))
ds = ds.batch(16, drop_remainder=True)

num_actions = 7
num_hidden_units = 2048
model = networks.InverseActionPolicy(num_actions, num_hidden_units)

cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001)

for epoch in range(0, 1000000):
    start_time = time.time()
    
    mean_loss = tf.keras.metrics.Mean()
    for idx, data in enumerate(ds):
        video, action, reward, done, N = data

        # video.shape:  (64, 20, 64, 64, 3)
        # action.shape:  (64, 20, 1)
        with tf.GradientTape() as tape:
            act_pi = model(video, training=True)
            act_pi = act_pi[:, :-1, :]

            act_dist = tfd.Categorical(logits=act_pi)
            pre_action = act_dist.sample()[0]

            action = tf.cast(tf.squeeze(action[:, :-1, :]), tf.int32)

            action_onehot = tf.one_hot(action, num_actions)
            
            print(" action[0]: ", action[0])
            print("pre_action: ", pre_action)

            act_loss = cce_loss(action_onehot, act_pi)

            regularization_loss = tf.reduce_sum(model.losses)

            total_loss = act_loss + 1e-5 * regularization_loss

            print('epoch: {}, idx: {}, total_loss: {}'.format(epoch, idx, total_loss))
            print("")
            
            mean_loss(total_loss)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
    if epoch % 20 == 0:
        model.save_weights("model/IDM_Model_{0}".format(epoch))