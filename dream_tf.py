import glob
import os
import math
import numpy as np
import random
import pathlib
from tqdm import tqdm
import time
import pygame, sys
import cv2
import gym
from pathlib import Path
from collections import defaultdict
from functools import partial

import utils
import networks
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

pygame.init()

width = 640
height = 640

gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Platyp us")
pygame.mouse.set_visible(True)

clock = pygame.time.Clock()


latent_dim = 512
num_embeddings = 512
tokenizer_tf = networks.VQ_VAE(latent_dim, num_embeddings)
tokenizer_tf.load_weights("model/CoinRun_VAVAQ_Model_60.ckpt")


env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=1, render_mode='rgb_array')


@tf.function
def dream_convert(observation):
    encoder_outputs = tokenizer_tf.encoder(observation)
    quantized_latents, _ = tokenizer_tf.vq_layer(encoder_outputs)
    reconstruction = tokenizer_tf.decoder(quantized_latents)
    #reconstruction = tf.clip_by_value(reconstruction, 0, 1)

    return reconstruction


for epoch in range(0, 1000):
    print("epoch: ", epoch)

    observation = env.reset() / 255.0

    for step in range(100000):
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

        observation, reward, done, info = env.step(action)
        observation = cv2.resize(observation, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        observation = observation / 255.0 - 0.5
        observation = np.expand_dims(observation, 0)
        #print("observation.shape: ", observation.shape)

        ######################################################
        #encoder_outputs = tokenizer_tf.encoder(observation)
        #quantized_latents, _ = tokenizer_tf.vq_layer(encoder_outputs)
        #reconstruction = tokenizer_tf.decoder(quantized_latents) + 0.5
        reconstruction = dream_convert(tf.constant(observation))
        ######################################################
        reconstruction = reconstruction.numpy()

        #print("observation.shape: ", observation.shape)
        #print("reconstruction.shape: ", reconstruction.shape)

        reconstruction = cv2.resize(reconstruction[0] * 255.0, dsize=(640, 640), interpolation=cv2.INTER_AREA)

        obs_surf = cv2.rotate(reconstruction, cv2.ROTATE_90_COUNTERCLOCKWISE)
        obs_surf = cv2.flip(obs_surf, 0)
        surf = pygame.surfarray.make_surface(obs_surf)
        gameDisplay.blit(surf, (0, 0))
        pygame.display.update()

        #if done:
        #    break