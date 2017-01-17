# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from preprocessing.vgg_preprocessing import unprocess_image
import utils
import os,sys

slim = tf.contrib.slim

def gram(feature):
  feature = tf.reshape(feature, tf.pack([-1, feature.get_shape()[3]]))
  return tf.matmul(feature, feature, transpose_a=True) / tf.to_float(tf.size(feature))

def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
      network_fn = nets_factory.get_network_fn(
          FLAGS.loss_model,
          num_classes=1,
          is_training=False)

      image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          FLAGS.loss_model,
          is_training=False)

      images = tf.expand_dims(utils.get_image(FLAGS.style_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
      _, endpoints_dict = network_fn(images)

      features = []
      for layer in FLAGS.style_layers:
          feature = endpoints_dict[layer]
          features.append(gram(feature))

      init_func = utils._get_init_fn(FLAGS)
      init_func(sess)
      if os.path.exists('generated') is False:
          os.makedirs('generated')
      save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
      with open(save_file, 'wb') as f:
          target_image = unprocess_image(images[0, :])
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

def pixel_loss(layer, FLAGS):
    generated_images, content_images = tf.split(0, 2, layer)

    #img_bytes = tf.read_file(FLAGS.mask_file)
    #maskimage = tf.image.decode_jpeg(img_bytes)
    #maskimage = tf.to_float(maskimage)
    #m_mean = tf.reduce_mean(maskimage, axis=(1,2))
    #index = tf.where(m_mean < 1.5)
    #top_index = index + tf.to_int64(1)
    #down_index = index - tf.to_int64(1)

    #select = tf.zeros_like(m_mean, dtype=tf.float32)
    #values = tf.squeeze(tf.ones_like(index, dtype=tf.float32))
    #topvalues = tf.squeeze(tf.ones_like(top_index, dtype=tf.float32))
    #downvalues = tf.squeeze(tf.ones_like(down_index, dtype=tf.float32))
    #delta = tf.SparseTensor(index, values, [FLAGS.image_size])
    #topdelta = tf.SparseTensor(index, topvalues, [FLAGS.image_size])
    #downdelta = tf.SparseTensor(index, downvalues, [FLAGS.image_size])
    #black_select = select + tf.sparse_tensor_to_dense(delta)
    #top_select = select + tf.sparse_tensor_to_dense(topdelta)
    #down_select = select + tf.sparse_tensor_to_dense(downdelta)

    #black_select = tf.mul(black_select, top_select)
    #black_select = tf.mul(black_select, down_select)
    #black_select = tf.expand_dims(black_select, -1)
    #black_select = tf.matmul(black_select, tf.ones([1, FLAGS.image_size]))
    #black_select = tf.expand_dims(black_select, -1)

    #generated_images = tf.mul(generated_images, black_select)
    #content_images = tf.mul(content_images, black_select)

    size = tf.size(generated_images)
    pixel_loss = tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)
    return pixel_loss

def content_loss(endpoints_dict, content_layers, layer_weights):
    content_loss = 0
    content_loss_summary = {}
    for index, layer in enumerate(content_layers):
        layer_w = layer_weights[index]
        generated_images, content_images = tf.split(0, 2, endpoints_dict[layer])
        size = tf.size(generated_images)
        layer_content_loss = tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
        content_loss_summary[layer] = layer_content_loss
        content_loss += layer_w * layer_content_loss
    return content_loss, content_loss_summary


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
