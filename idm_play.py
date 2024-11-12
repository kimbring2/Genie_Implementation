import numpy as np
import random
import os
import gym
import cv2
import time

env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=1, render_mode='rgb_array')
#print("env.action_space: ", env.action_space)

max_trials = 20000

action_list = [1, 2, 3, 4, 5, 7, 8]
pre_action_list = [2, 3, 2, 3, 3, 5, 5, 5, 5, 6, 4, 5, 6, 3, 5, 2, 5, 5, 3, 2, 2, 5, 4, 5, 4, 4, 6, 4, 2, 3, 6, 2, 5, 6, 6, 2, 2, 5, 5, 1]

for trial in range(max_trials):
  tot_r = 0
  obs = env.reset() / 255.0

  for step in range(0, len(pre_action_list)):
    action = pre_action_list[step]

    action = action_list[pre_action_list[step]]
    obs, reward, done, info = env.step(action)

    frame = cv2.resize(obs, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    time.sleep(0.1)