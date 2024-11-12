import tensorflow as tf
import numpy as np
import math
import einops
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Embedding, Layer, BatchNormalization, Normalization, ReLU, Add
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from typing import Any, List, Sequence, Tuple


class ResizeVideo(tf.keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)

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
    
  def call(self, state: tf.Tensor, training) -> Tuple[tf.Tensor]:
    state = tf.cast(state, tf.float32)

    batch_size = state.shape[0]
    time_step = state.shape[1]

    conv3d = self.conv3d(state)
    conv3d = tf.keras.layers.LayerNormalization()(conv3d)
    conv3d = tf.keras.layers.ReLU()(conv3d)

    conv3d_reshaped = tf.reshape(conv3d, [batch_size * time_step, *conv3d.shape[-3:]])

    paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])

    conv3d_padded = tf.pad(conv3d_reshaped, paddings, "CONSTANT")
    conv2d_1 = self.conv2d_1(conv3d_padded)
    conv2d_1 = tf.pad(conv2d_1, paddings, "CONSTANT")
    conv2d_1 = self.max_pool_2d_1(conv2d_1)

    conv2d_1 = tf.pad(conv2d_1, paddings, "CONSTANT")
    conv2d_2 = self.conv2d_2(conv2d_1)
    conv2d_2 = tf.pad(conv2d_2, paddings, "CONSTANT")
    conv2d_2 = self.max_pool_2d_2(conv2d_2)

    conv2d_2 = tf.pad(conv2d_2, paddings, "CONSTANT")
    conv2d_3 = self.conv2d_3(conv2d_2)
    conv2d_3 = tf.pad(conv2d_3, paddings, "CONSTANT")
    conv2d_3 = self.max_pool_2d_3(conv2d_3)

    conv2d_3_reshaped = tf.reshape(conv2d_3, [batch_size, time_step, *conv2d_3.shape[1:]])

    conv2d_3_flanttned = tf.reshape(conv2d_3_reshaped, [batch_size, time_step, -1])
      
    X_input = self.common(conv2d_3_flanttned)
      
    pi_latent  = self.actor(X_input)

    return pi_latent


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.beta = beta

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True, name="embeddings_vqvae",
        )

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distance_1 = tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
        distance_2 = tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity
        distances = distance_1 + distance_2
        encoding_indices = tf.argmin(distances, axis=1)
        
        return encoding_indices

    def call(self, x):
        input_shape = tf.shape(x)
        
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)
        encoding_indices_reshaped = tf.reshape(encoding_indices, [-1, 4 * 4])
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        
        return quantized, encoding_indices_reshaped


class VQ_VAE(tf.keras.Model):
  def __init__(self, latent_dim, num_embeddings):
    super(VQ_VAE, self).__init__()

    self.latent_dim = latent_dim
    self.num_embeddings = num_embeddings

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(128, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(256, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2D(latent_dim, 1, padding="same")
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(4, 4, 512)),
            tf.keras.layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            tf.keras.layers.Conv2DTranspose(3, 3, padding="same")
        ]
    )

    self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")

  def call(self, inputs): 
    encoder_outputs = self.encoder(inputs)
    quantized_latents, encoding_indices_reshaped = self.vq_layer(encoder_outputs)
    reconstructions = self.decoder(quantized_latents)

    return reconstructions, encoding_indices_reshaped


class Slicer():
    def __init__(self, max_blocks, block_mask):
        super().__init__()

        self.block_size = block_mask.shape[0]
        self.num_kept_tokens = tf.reduce_sum(block_mask)
        self.num_kept_tokens = tf.cast(self.num_kept_tokens, tf.int32)

        kept_indices = tf.where(block_mask)
        kept_indices = tf.reshape(kept_indices, [-1])

        kept_indices = tf.tile(kept_indices, tf.constant([max_blocks], tf.int32))
        kept_indices = tf.cast(kept_indices, tf.int32)

        offsets = tf.range(max_blocks)
        offsets = tf.repeat(offsets, repeats=self.num_kept_tokens)
        self.indices = kept_indices + block_mask.shape[0] * offsets

    def compute_slice(self, num_steps, prev_steps):
        total_steps = num_steps + prev_steps
        num_blocks = tf.math.ceil(total_steps / self.block_size)
        num_blocks = tf.cast(num_blocks, tf.int32)
        indices = self.indices[:num_blocks * self.num_kept_tokens]

        result = indices[tf.math.logical_and(prev_steps <= indices, indices < total_steps)]
        result = result - prev_steps

        return result


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, max_tokens, d_model, num_heads, dropout):
    super(MultiHeadAttention, self).__init__()

    self.max_tokens = max_tokens
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.layernorm = tf.keras.layers.LayerNormalization()
    self.dropout_1 = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(d_model)
    self.dropout_2 = tf.keras.layers.Dropout(dropout)

    self.mask = tf.experimental.numpy.tril(tf.ones((max_tokens, max_tokens)))
    self.mask = tf.ones_like(self.mask) - self.mask

  def get_config(self):
    config = super().get_config().copy()
    config.update({'shape': self.shape, 'd_model': self.d_model, 'num_heads': self.num_heads})
    return config
    
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, q, k, v, decode_step, k_cache=None, v_cache=None):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
      
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    if decode_step is not None:
        k = tf.transpose(k, [2, 0, 1, 3])
        v = tf.transpose(v, [2, 0, 1, 3])

        k_cache = tf.tensor_scatter_nd_update(k_cache, [[decode_step]], k)
        v_cache = tf.tensor_scatter_nd_update(v_cache, [[decode_step]], v)

        k = k_cache
        k = tf.transpose(k, [1, 2, 0, 3])

        v = v_cache
        v = tf.transpose(v, [1, 2, 0, 3])
      
    mask = self.mask
    if decode_step is not None and self.mask is not None:
        mask = mask[decode_step]

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)
        
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    attention_weights = self.dropout_1(attention_weights)
      
    scaled_attention = tf.matmul(attention_weights, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    #scaled_attention = tf.reshape(scaled_attention, [1, 1, 256])

    if decode_step is None:
        scaled_attention = tf.reshape(scaled_attention, [scaled_attention.shape[0], scaled_attention.shape[1], scaled_attention.shape[2]*scaled_attention.shape[3]])
        scaled_attention = tf.reshape(scaled_attention, [batch_size, self.max_tokens, self.d_model])
    else:
        scaled_attention = tf.reshape(scaled_attention, [batch_size, 1, self.d_model])
      
    output = self.dense(scaled_attention)
    output = self.dropout_2(output)

    return output, attention_weights, k_cache, v_cache


class AttentionBlock(tf.keras.layers.Layer):
  def __init__(self, max_tokens, num_heads, key_dim, dropout):
    super().__init__()

    self.max_tokens = max_tokens
    self.num_heads = num_heads
      
    self.layernorm_1 = tf.keras.layers.LayerNormalization()
    self.mha = MultiHeadAttention(max_tokens, key_dim, num_heads, dropout)
    self.post_attn_dp = tf.keras.layers.Dropout(rate=dropout)
    self.add_1 = tf.keras.layers.Add()

    self.fc_block = tf.keras.Sequential([tf.keras.layers.Dense(units=4*key_dim, activation='gelu'), tf.keras.layers.Dense(units=key_dim)])

    self.layernorm_2 = tf.keras.layers.LayerNormalization()
    self.post_fc_dp = tf.keras.layers.Dropout(rate=dropout)
    self.add_2 = tf.keras.layers.Add()

  def call(self, x, decode_step=None, k_cache=None, v_cache=None):
    h = self.layernorm_1(x)
      
    h, _, k_cache, v_cache = self.mha(h, h, h, decode_step, k_cache, v_cache)

    h = self.post_attn_dp(h)
    x = self.add_1([x, h])

    h = self.layernorm_2(x)
    h = self.fc_block(h)
    h = self.post_fc_dp(h)
      
    x = self.add_2([x, h])
    
    return x, k_cache, v_cache



class AttentionStack(tf.keras.layers.Layer):
  def __init__(self, max_tokens, units, num_heads=1, num_layers=2, dropout_rate=0.1):
    super().__init__()

    self.max_tokens = max_tokens
    self.units = units
      
    self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
    self.attn_blocks = [AttentionBlock(max_tokens=max_tokens, num_heads=num_heads, key_dim=units, dropout=dropout_rate) for n in range(num_layers)]
    self.layernorm_1 = tf.keras.layers.LayerNormalization()
      
  def call(self, inputs, decode_step=None, k_cache_array=None, v_cache_array=None, training=False):
    inputs_shape = inputs.shape

    inputs = self.dropout_1(inputs)
      
    if decode_step is None:
        for attn_blocks in self.attn_blocks:
            inputs, _, _ = attn_blocks(inputs, decode_step)

        k_cache_array_new = k_cache_array
        v_cache_array_new = v_cache_array
    else:
        k_cache_array_new = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        v_cache_array_new = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for idx, attn_blocks in enumerate(self.attn_blocks):
            k_cache = k_cache_array[idx]
            v_cache = v_cache_array[idx]

            inputs, k_cache_new, v_cache_new = attn_blocks(inputs, decode_step, k_cache, v_cache)

            k_cache_array_new = k_cache_array_new.write(idx, k_cache_new)
            v_cache_array_new = v_cache_array_new.write(idx, v_cache_new)

        k_cache_array_new = k_cache_array_new.stack()
        v_cache_array_new = v_cache_array_new.stack()

    inputs = self.layernorm_1(inputs)
      
    return inputs, k_cache_array_new, v_cache_array_new



class WorldModel(tf.keras.Model):
    def __init__(self, max_blocks, tokens_per_block, max_tokens, obs_vocab_size, act_vocab_size, 
                 num_layers=1, units=256, num_heads=1, dropout_rate=0.1, obs_slicer=None, act_slicer=None, head_obs_slicer=None):
        super().__init__()

        self.max_blocks = max_blocks
        self.tokens_per_block = tokens_per_block
        self.max_tokens = max_tokens
        self.obs_vocab_size = obs_vocab_size
        self.act_vocab_size = act_vocab_size
        self.num_layers = num_layers
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
      
        self.attn_stack = AttentionStack(max_tokens, units, num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate)
        
        self.obs_slicer = obs_slicer
        self.act_slicer = act_slicer
        self.head_obs_slicer = head_obs_slicer
        
        self.pos_emb = Embedding(max_tokens, units, embeddings_initializer=RandomNormal(mean=0.0, stddev=0.02))
        self.embedding_table_obs = Embedding(obs_vocab_size, units, embeddings_initializer=RandomNormal(mean=0.0, stddev=0.02))
        self.embedding_table_act = Embedding(act_vocab_size, units, embeddings_initializer=RandomNormal(mean=0.0, stddev=0.02))

        self.head_observations = tf.keras.Sequential([tf.keras.layers.Dense(units=units, activation='relu'), tf.keras.layers.Dense(units=obs_vocab_size)])
        #self.head_rewards = tf.keras.Sequential([tf.keras.layers.Dense(units=units, activation='relu'), tf.keras.layers.Dense(units=3)])
        self.head_ends = tf.keras.Sequential([tf.keras.layers.Dense(units=units, activation='relu'), tf.keras.layers.Dense(units=2)])
        
    def call(self, tokens, decode_step=None, k_cache_array=None, v_cache_array=None):
        batch_size = tokens.shape[0]
        token_size = tokens.shape[1]

        if decode_step == None:
            prev_steps = 0
            num_steps = self.max_tokens
        else:
            prev_steps = decode_step
            num_steps = 1
        
        obs_slice = self.obs_slicer.compute_slice(num_steps, prev_steps)
        act_slice = self.act_slicer.compute_slice(num_steps, prev_steps)
        end_slice = self.act_slicer.compute_slice(num_steps, prev_steps)
        head_obs_slice = self.head_obs_slicer.compute_slice(num_steps, prev_steps)
        
        if decode_step == None:
            x = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            for i in tf.range(0, batch_size):
                output_batch = tf.zeros((token_size, 256), dtype=tf.float32)
                
                token = tokens[i]
                
                token_obs = tf.gather_nd(token, tf.expand_dims(obs_slice, 1))
                token_act = tf.gather_nd(token, tf.expand_dims(act_slice, 1))

                token_obs_emb = self.embedding_table_obs(token_obs)
                token_act_emb = self.embedding_table_act(token_act)

                output_batch = tf.tensor_scatter_nd_update(output_batch, tf.expand_dims(act_slice, 1), token_act_emb, name=None)
                output_batch = tf.tensor_scatter_nd_update(output_batch, tf.expand_dims(obs_slice, 1), token_obs_emb, name=None)                                     
            
                x = x.write(i, output_batch)

            x = x.stack()
        
            emb = self.pos_emb(prev_steps + np.arange(num_steps))
            x = x + emb
        else:
            emb = self.pos_emb(prev_steps + tf.range(num_steps))
            x = tokens + emb

        logits, k_cache_array, v_cache_array = self.attn_stack(x, decode_step, k_cache_array, v_cache_array)

        logits_obs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        logits_rew = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        logits_end = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            logit = logits[i]

            logit_obs = tf.gather_nd(logit, tf.expand_dims(head_obs_slice, 1))
            logit_rew = tf.gather_nd(logit, tf.expand_dims(act_slice, 1))
            logit_end = tf.gather_nd(logit, tf.expand_dims(act_slice, 1))

            logits_obs = logits_obs.write(i, logit_obs)
            logits_rew = logits_rew.write(i, logit_rew)
            logits_end = logits_end.write(i, logit_end)
        
        logits_obs = logits_obs.stack()
        logits_rew = logits_rew.stack()
        logits_end = logits_end.stack()
    
        logits_obs = self.head_observations(logits_obs)
        #logits_rew = self.head_rewards(logits_rew)
        logits_end = self.head_ends(logits_end)

        return logits_obs, 0, logits_end, k_cache_array, v_cache_array