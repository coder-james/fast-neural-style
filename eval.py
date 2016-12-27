#!/usr/bin/python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import transform_model
import time
import os

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models/chritmas500/fast-style-model.ckpt-done", "")
tf.app.flags.DEFINE_string("image_file", "img/test.jpg", "")

FLAGS = tf.app.flags.FLAGS


def main(_):
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = utils.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
            image = tf.expand_dims(image, 0)
            generated = transform_model.net(image, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            start_time = time.time()
            generated = sess.run(generated)
            generated = tf.cast(generated, tf.uint8)
            end_time = time.time()
            tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
