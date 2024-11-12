import numpy as np
import os
import cv2
import random
import time
import math
import pathlib
import pygame, sys
import gym
from typing import Any, List, Sequence, Tuple

pygame.init()

width = 640
height = 640

gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Platyp us")
pygame.mouse.set_visible(True)

clock = pygame.time.Clock()

import tensorflow as tf
import tensorflow_probability as tfp

import utils
import networks

tfd = tfp.distributions

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


'''
class InverseActionPolicy(tf.keras.Model):
  """Inverse Dynamics  network."""
  def __init__(self, num_actions: int, num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions

    # obs
    self.conv3d = tf.keras.layers.Conv3D(filters=128, kernel_size=(5, 1, 1), padding="same")

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
model.load_weights("model/IDM_Model_160")

env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=1, render_mode='rgb_array')
action_list = [1, 2, 3, 4, 5, 7, 8]


for epoch in range(0, 1000):
    print("epoch: ", epoch)

    video = []

    obs = env.reset()
    video.append(obs / 255.0)

    last_obs = obs

    for step in range(100000):
        if len(video) == 20:
            video_array = np.array(video)
            print("video_array.shape: ", video_array.shape)

            act_pi = model(tf.expand_dims(video_array, 0), training=False)
            print("act_pi.shape: ", act_pi.shape)

            act_dist = tfd.Categorical(logits=act_pi)
            pre_action = act_dist.sample()[0].numpy()
            print("pre_action.tolist(): ", pre_action.tolist())

            video = []
            video.append(last_obs / 255.0)
            print("")

        #print("step: ", step)

        pygame.event.set_grab(True)

        exit = False
        action = 4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

        pressed_keys = pygame.key.get_pressed()
        #print("pressed_keys: ", pressed_keys)

        if pressed_keys[pygame.K_w]:
            action = 5

        if pressed_keys[pygame.K_a]:
            action = 1

        if pressed_keys[pygame.K_w] & pressed_keys[pygame.K_a]:
            action = 2

        if pressed_keys[pygame.K_s]:
            action = 3

        if pressed_keys[pygame.K_d]:
            action = 7

        if pressed_keys[pygame.K_w] & pressed_keys[pygame.K_d]:
            action = 8

        #action = types_np.sample(env.ac_space, bshape=(env.num,))
        #print("type(action): ", type(action))
        #print("action: ", action)

        obs, reward, done, info = env.step(action)
        video.append(obs / 255.0)
        last_obs = obs

        #frame = np.concatenate((obs[0], recon[0]), 1)
        reconstruction = cv2.resize(obs, dsize=(640, 640), interpolation=cv2.INTER_AREA)

        obs_surf = cv2.rotate(reconstruction, cv2.ROTATE_90_COUNTERCLOCKWISE)
        obs_surf = cv2.flip(obs_surf, 0)
        surf = pygame.surfarray.make_surface(obs_surf)
        gameDisplay.blit(surf, (0, 0))
        pygame.display.update()

        if done:
            break

        time.sleep(0.1)