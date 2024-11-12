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
from pathlib import Path
from collections import defaultdict
from functools import partial
import sys

import tensorflow as tf
import tensorflow_probability as tfp
import utils
import networks

tfd = tfp.distributions

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
world_model.load_weights("model/world_model_2")


@tf.function
def env_reset(obs_tokens, k_cache_array, v_cache_array):
    initial_k_cache_shape = k_cache_array.shape
    initial_v_cache_shape = v_cache_array.shape

    batch_size = obs_tokens.shape[0]
    for i in tf.range(0, 16):
        obs_token = obs_tokens[i]
        obs_token = tf.reshape(obs_token, [1, 1])
        obs_token_emb = world_model.embedding_table_obs(obs_token)

        logit_obs, _, _, k_cache_array, v_cache_array = world_model(obs_token_emb, decode_step=i, k_cache_array=k_cache_array, v_cache_array=v_cache_array)

        k_cache_array.set_shape(initial_k_cache_shape)
        v_cache_array.set_shape(initial_v_cache_shape)

    return k_cache_array, v_cache_array


@tf.function
def env_step(act_token, k_cache_array, v_cache_array, decode_step):
    batch_size = act_token.shape[0]

    obs_tokens = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    act_token_emb = world_model.embedding_table_act(act_token)
    logit_obs, _, _, k_cache_array, v_cache_array = world_model(act_token_emb, decode_step=decode_step, k_cache_array=k_cache_array, v_cache_array=v_cache_array)
    logit_obs = tf.squeeze(logit_obs)
    dist = tfd.Categorical(logits=logit_obs)
    sample = dist.sample()
    obs_token = tf.reshape(sample, [1, 1])

    obs_tokens = obs_tokens.write(0, obs_token)

    token_add_step = 1
    for i in tf.range(0, tokens_per_block - 1):
        obs_token_emb = world_model.embedding_table_obs(obs_token)

        logit_obs, _, _, k_cache_array, v_cache_array = world_model(obs_token_emb, decode_step=decode_step + 1 + i, k_cache_array=k_cache_array, v_cache_array=v_cache_array)
        logit_obs = tf.squeeze(logit_obs)

        if i < tokens_per_block - 2:
            dist = tfd.Categorical(logits=logit_obs)
            sample = dist.sample()
            obs_token = tf.reshape(sample, [1, 1])
            obs_tokens = obs_tokens.write(i + 1, obs_token)
            token_add_step += 1

    obs_tokens = obs_tokens.stack()
    obs_tokens = tf.squeeze(obs_tokens)
    encodings = tf.one_hot(obs_tokens, tokenizer_tf.vq_layer.num_embeddings)
    quantized = tf.matmul(encodings, tokenizer_tf.vq_layer.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, [1, 4, 4, 512])
    reconstructions = tokenizer_tf.decoder(quantized)[0]
    reconstructions = tf.clip_by_value(reconstructions, -0.5, 0.5)

    return reconstructions, k_cache_array, v_cache_array, obs_tokens


for epoch in range(0, 1000):
    print("epoch: ", epoch)

    decode_step = 0
    obs_tokens = np.array([173, 150, 335,  66, 173, 390, 238, 134, 173, 173, 173, 134, 509, 509, 509, 134])
    action_list = [1, 2, 3, 4, 5, 7, 8]

    k_shape = (num_layers, max_tokens, 1, 4, 64)
    v_shape = (num_layers, max_tokens, 1, 4, 64)
    k_cache_array = tf.zeros(k_shape, dtype=tf.float32)
    v_cache_array = tf.zeros(v_shape, dtype=tf.float32)
    k_cache_array, v_cache_array = env_reset(obs_tokens, k_cache_array, v_cache_array)
    decode_step += 16

    for step in range(100000):
        print("step: ", step)

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

        action_token = action_list.index(action)

        #print("action_token: ", action_token)
        act_token = tf.constant([[action_token]])
        reconstructions, k_cache_array, v_cache_array, obs_tokens = env_step(act_token, k_cache_array, v_cache_array, decode_step)
        print("obs_tokens: ", obs_tokens)
        decode_step += 17

        if decode_step + 17 >= 340:
            decode_step = 0
            k_cache_array = tf.zeros(k_shape, dtype=tf.float32)
            v_cache_array = tf.zeros(v_shape, dtype=tf.float32)
            k_cache_array, v_cache_array = env_reset(obs_tokens, k_cache_array, v_cache_array)
            decode_step += 16

        reconstructions = reconstructions.numpy()
        reconstructions = cv2.resize(reconstructions, dsize=(640, 640), interpolation=cv2.INTER_AREA)

        obs_surf = cv2.rotate(reconstructions * 255.0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        obs_surf = cv2.flip(obs_surf, 0)
        surf = pygame.surfarray.make_surface(obs_surf)
        gameDisplay.blit(surf, (0, 0))
        pygame.display.update()

        #if done:
        #    break