import numpy as np
import random
import os
import gym
import cv2

env = gym.make('procgen:procgen-coinrun-v0', start_level=10, num_levels=0, render_mode='rgb_array')
print("env.action_space: ", env.action_space)

dir_name = 'record_multi'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

max_trials = 20000
min_frames = 200
action_list = [1, 2, 3, 4, 5, 7, 8]

total_frames = 0
for trial in range(max_trials):
  random_generated_int = random.randint(0, 2**31 - 1)
  filename = dir_name + "/" + str(random_generated_int) + ".npz"

  recording_N = []
  recording_frame = []
  recording_action = []
  recording_reward = []
  recording_done = []

  np.random.seed(random_generated_int)

  tot_r = 0
  obs = env.reset() / 255.0

  step = 0
  while True:
    recording_N.append(1)
    recording_frame.append(obs)
    
    #action = env.action_space.sample()
    action_index = random.randint(0, len(action_list) - 1)
    action = action_list[action_index]
    #print("action: ", action)
    recording_action.append(action_index)

    obs, reward, done, info = env.step(action)
    obs = obs / 255.0
    #print("obs.shape: ", obs.shape)
    tot_r += reward

    recording_reward.append(reward)
    recording_done.append(done)

    #frame = cv2.resize(obs, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    #cv2.imshow('frame', frame)
    #cv2.waitKey(1)

    if done:
      print('total reward {}'.format(tot_r))
      break

    step += 1

  total_frames += (step + 1)
  print('total reward {}'.format(tot_r))
  print("dead at", step + 1, "total recorded frames for this worker", total_frames)

  recording_frame = np.array(recording_frame, dtype=np.float32)
  recording_action = np.array(recording_action, dtype=np.float16)
  recording_reward = np.array(recording_reward, dtype=np.float16)
  recording_done = np.array(recording_done, dtype=np.bool)
  recording_N = np.array(recording_N, dtype=np.uint16)
  
  if len(recording_frame) > min_frames:
    np.savez_compressed(filename, obs=recording_frame, action=recording_action, reward=recording_reward, done=recording_done, N=recording_N)