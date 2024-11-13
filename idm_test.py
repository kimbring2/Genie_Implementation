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


num_actions = 7
num_hidden_units = 2048
model = networks.InverseActionPolicy(num_actions, num_hidden_units)
model.load_weights("model/IDM_Model_80")

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