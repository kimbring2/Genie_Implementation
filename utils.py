import tensorflow as tf


def masked_fill(tensor, mask, value):
    return tf.where(mask, tf.fill(tf.shape(tensor), value), tensor)


def compute_labels_world_model(obs_tokens, rewards, ends):
    batch_size = obs_tokens.shape[0]

    #mask_fill = tf.math.logical_not(mask_padding)

    #labels_observations = masked_fill(obs_tokens, tf.tile(tf.expand_dims(mask_fill, -1), tf.constant([1,1,16], tf.int32)), -100)
    labels_observations = obs_tokens
    labels_observations = tf.reshape(labels_observations, [batch_size, -1])
    labels_observations = labels_observations[:, 1:]

    #labels_rewards = masked_fill((rewards + 1), mask_fill, -100)
    labels_rewards = rewards
    labels_rewards = tf.reshape(labels_rewards, -1)
    
    #labels_ends = masked_fill(ends, mask_fill, -100)
    labels_ends = ends
    labels_ends = tf.reshape(labels_ends, -1)
    
    return tf.reshape(labels_observations, -1), tf.reshape(labels_rewards, -1), tf.reshape(labels_ends, -1)