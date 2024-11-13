import glob
import os
import math
import numpy as np
import random
import pathlib
import time
import pygame, sys
import cv2
import gym


pygame.init()

width = 640
height = 640

gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Platyp us")
pygame.mouse.set_visible(True)

clock = pygame.time.Clock()

env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=0, render_mode='rgb_array')
#env = gym.make('procgen:procgen-coinrun-v0', start_level=10, num_levels=0, render_mode='rgb_array')
print("env.action_space: ", env.action_space)


for epoch in range(0, 1000):
    print("epoch: ", epoch)

    obs = env.reset()

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

        #action = types_np.sample(env.ac_space, bshape=(env.num,))
        #print("type(action): ", type(action))
        print("action: ", action)

        obs, reward, done, info = env.step(action)

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