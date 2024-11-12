import numpy as np
import os
import cv2
import json
import tensorflow as tf
import random
from vq_vae import VQ_VAE

from utils import PARSER


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

args = PARSER.parse_args()


DATA_DIR = "results/record"
SERIES_DIR = "results/series"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)


def ds_gen():
    filenames = os.listdir(DATA_DIR)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue

        file_path = os.path.join(DATA_DIR, fname)
        try:
            data = np.load(file_path)
            img = data['obs']

            action = np.reshape(data['action'], newshape=[-1, args.a_width])
            reward = data['reward']
            done = data['done']
            N = data['N']
            
            n_pad = args.max_frames - img.shape[0] # pad so they are all a thousand step long episodes
            img = tf.pad(img, [[0, n_pad], [0, 0], [0, 0], [0, 0]], constant_values=-100.0/255.0)
            action = tf.pad(action, [[0, n_pad], [0, 0]])
            reward = tf.pad(reward, [[0, n_pad]], constant_values=-100.0)
            done = tf.pad(done, [[0, n_pad]], constant_values=True)

            N = tf.pad(N, [[0, n_pad]], constant_values=0)

            yield img, action, reward, done, N
        except:
            print("npz file load fail: {}".format(file_path))


def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32, tf.float32, tf.bool, tf.int16), 
                output_shapes=((args.max_frames, 64, 64, 3), (args.max_frames, args.a_width), (args.max_frames,), (args.max_frames,), (args.max_frames,)))
    
    return dataset


#@tf.function
def encode_batch(batch_img):
  batch_img -= 0.5
  encoder_outputs = vqvae.encoder(batch_img)
  #quantized_latents = vqvae.vq_layer(encoder_outputs)
  quantized_latents, encoding_indices_reshaped  = vqvae.vq_layer(encoder_outputs)
  #print("quantized_latents.shape: ", quantized_latents.shape)

  return encoding_indices_reshaped.numpy()


def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 1)
  
  return batch_img


filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_tf_dataset()
dataset = dataset.batch(1, drop_remainder=True)

latent_dim = 16
num_embeddings = 128
vqvae = VQ_VAE(latent_dim, num_embeddings)
vqvae.load_weights("model/VizDoom_CVAE_Model_170.ckpt")


quantized_latents_dataset = []
action_dataset = []
r_dataset = []
d_dataset = []
N_dataset = []

epoch = 0
i = 0
for batch in dataset:
  i += 1
  obs_batch, action_batch, r, d, N = batch
  obs_batch = tf.squeeze(obs_batch, axis=0)
  action_batch = tf.squeeze(action_batch, axis=0)
  r = tf.reshape(r, [-1, 1])
  d = tf.reshape(d, [-1, 1])
  N = tf.reshape(N, [-1, 1])

  quantized_latents = encode_batch(obs_batch)
  quantized_latents_dataset.append(quantized_latents.astype(np.float32))
  action_dataset.append(action_batch.numpy())
  r_dataset.append(r.numpy().astype(np.float16))
  d_dataset.append(d.numpy().astype(np.bool))
  N_dataset.append(N.numpy().astype(np.uint16))

  if (i + 1) % 100 == 0:
    print(i + 1)

#if (i + 1) % 10000 == 0:
quantized_latents_dataset = np.array(quantized_latents_dataset)
action_dataset = np.array(action_dataset)
r_dataset = np.array(r_dataset)
d_dataset = np.array(d_dataset)
N_dataset = np.array(N_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz".format(epoch)), action=action_dataset, quantized_latents=quantized_latents_dataset, 
                    reward=r_dataset, done=d_dataset, N=N_dataset)
'''
quantized_latents_dataset = []
action_dataset = []
r_dataset = []
d_dataset = []
N_dataset = []

epoch += 1
'''