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

'''
class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t = old_shape['t'])

    return videos

class InverseActionPolicy(tf.keras.Model):
  def __init__(self, num_actions: int, num_hidden_units: int):
    super().__init__()

    self.num_actions = num_actions

    self.conv3d = tf.keras.layers.Conv3D(filters=128, kernel_size=(5, 1, 1), padding="same")
    self.resize_video = ResizeVideo(32, 32)

    self.conv2d_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), activation='relu')
    self.max_pool_2d_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

    self.conv2d_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu')
    self.max_pool_2d_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

    self.conv2d_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu')
    self.max_pool_2d_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

    self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True, kernel_regularizer='l2')
    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')

    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({'num_actions': self.num_actions, 'num_hidden_units': self.num_hidden_units})

    return config
    
  def call(self, state: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    #print("state.shape: ", state.shape)

    state = tf.cast(state, tf.float32)

    batch_size = state.shape[0]
    time_step = state.shape[1]

    #state = self.resize_video(state)

    conv3d = self.conv3d(state)
    conv3d = tf.keras.layers.LayerNormalization()(conv3d)
    conv3d = tf.keras.layers.ReLU()(conv3d)
    # conv3d.shape:  (2, 20, 32, 32, 128)
    #print("conv3d.shape: ", conv3d.shape)

    conv3d_reshaped = tf.reshape(conv3d, [batch_size * time_step, *conv3d.shape[-3:]])
    # conv3d_reshaped.shape:  (40, 64, 64, 128)

    paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])

    conv3d_padded = tf.pad(conv3d_reshaped, paddings, "CONSTANT")
    conv2d_1 = self.conv2d_1(conv3d_padded)
    conv2d_1 = tf.pad(conv2d_1, paddings, "CONSTANT")
    conv2d_1 = self.max_pool_2d_1(conv2d_1)
    #print("conv2d_1.shape: ", conv2d_1.shape)

    conv2d_1 = tf.pad(conv2d_1, paddings, "CONSTANT")
    conv2d_2 = self.conv2d_2(conv2d_1)
    conv2d_2 = tf.pad(conv2d_2, paddings, "CONSTANT")
    conv2d_2 = self.max_pool_2d_2(conv2d_2)
    #print("conv2d_2.shape: ", conv2d_2.shape)

    conv2d_2 = tf.pad(conv2d_2, paddings, "CONSTANT")
    conv2d_3 = self.conv2d_3(conv2d_2)
    conv2d_3 = tf.pad(conv2d_3, paddings, "CONSTANT")
    conv2d_3 = self.max_pool_2d_3(conv2d_3)
    #print("conv2d_3.shape: ", conv2d_3.shape)

    conv2d_3_reshaped = tf.reshape(conv2d_3, [batch_size, time_step, *conv2d_3.shape[1:]])
    #print("conv2d_3_reshaped.shape: ", conv2d_3_reshaped.shape)

    conv2d_3_flanttned = tf.reshape(conv2d_3_reshaped, [batch_size, time_step, -1])
    #print("conv2d_3_flanttned.shape: ", conv2d_3_flanttned.shape)

    #initial_state = (memory_state, carry_state)
    #lstm_output, final_memory_state, final_carry_state  = self.lstm(conv3d_reshaped, initial_state=initial_state, training=training)
    #print("lstm_output.shape: ", lstm_output.shape)
      
    X_input = self.common(conv2d_3_flanttned)
    #print("X_input: ", X_input)
      
    pi_latent  = self.actor(X_input)
    #print("pi_latent.shape: ", pi_latent.shape)
    #print("")

    return pi_latent, memory_state, carry_state
'''

num_actions = 7
num_hidden_units = 2048
model = networks.InverseActionPolicy(num_actions, num_hidden_units)

cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.0001)

for epoch in range(0, 1000000):
    start_time = time.time()
    
    mean_loss = tf.keras.metrics.Mean()
    for idx, data in enumerate(ds):
        #print("idx: ", idx)
        video, action, reward, done, N = data

        # video.shape:  (64, 20, 64, 64, 3)
        # action.shape:  (64, 20, 1)
        with tf.GradientTape() as tape:
            act_pi = model(video, training=True)
            
            #act_pi = prediction[0]
            # act_pi.shape:  (64, 20, 7)
            act_pi = act_pi[:, :-1, :]
            #print("act_pi.shape: ", act_pi.shape)

            act_dist = tfd.Categorical(logits=act_pi)
            pre_action = act_dist.sample()[0]

            action = tf.cast(tf.squeeze(action[:, :-1, :]), tf.int32)
            #print("action.shape: ", action.shape)

            action_onehot = tf.one_hot(action, num_actions)
            
            #print("action_onehot.shape: ", action_onehot.shape)
            #print("act_pi[0].shape: ", act_pi[0].shape)

            print(" action[0]: ", action[0])
            print("pre_action: ", pre_action)

            #print("action_onehot.shape: ", action_onehot.shape)
            #print("act_pi.shape: ", act_pi.shape)
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