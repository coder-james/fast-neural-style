# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import os,sys

slim = tf.contrib.slim

def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default(), tf.Session() as sess:
      network_fn = nets_factory.get_network_fn(
          FLAGS.loss_model,
          num_classes=1,
          is_training=False)

      image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
          FLAGS.loss_model,
          is_training=False)

      images = tf.expand_dims(utils.get_image(FLAGS.style_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
      _, endpoints_dict = network_fn(images, spatial_squeeze=False)

      features = []
      for layer in FLAGS.style_layers:
          feature = endpoints_dict[layer]
          feature = tf.reshape(feature, tf.pack([-1, feature.get_shape()[3]]))
          feature = tf.matmul(feature, feature, transpose_a=True) / tf.to_float(tf.size(feature))
          features.append(feature)

      init_func = utils._get_init_fn(FLAGS)
      init_func(sess)
      if os.path.exists('generated') is False:
          os.makedirs('generated')
      save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
      with open(save_file, 'wb') as f:
          target_image = image_unprocessing_fn(images[0, :])
          value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
          f.write(sess.run(value))
          tf.logging.info('Target style pattern is saved to: %s.' % save_file)
      return sess.run(features)


def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(0, 2, endpoints_dict[layer])
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(0, 2, endpoints_dict[layer])
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
