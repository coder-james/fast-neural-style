#!/usr/bin/python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from PIL import Image
import numpy as np
import utils
import transform_model, sr_model, al_model
import time
import os

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models/alipay_mask_pixel/fast-alipay-model.ckpt-done", "")
tf.app.flags.DEFINE_string("image_file", "img/lbj.jpg", "")
tf.app.flags.DEFINE_string("model_type", "alipay", "model used (super/transform/alipay)")
tf.app.flags.DEFINE_integer("image_scale", 4, "image scale to process")
tf.app.flags.DEFINE_string("same_shape", False, "whether resize to origin shape or not")

FLAGS = tf.app.flags.FLAGS

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    image = Image.open(FLAGS.image_file)
    image = np.asarray(image)
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    tf.logging.info('Image size: %dx%dx%d' % (width, height, channel))

    with tf.Graph().as_default():
        with tf.Session(config=config).as_default() as sess:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            rawimage = utils.get_image(FLAGS.image_file, 256, 256, image_preprocessing_fn)
            rawimage = tf.expand_dims(rawimage, 0)
            rawimage = tf.to_float(rawimage)
            if FLAGS.model_type == "transform":
              generated = transform_model.net(rawimage, training=False)
            elif FLAGS.model_type == "super":
              generated = sr_model.net(rawimage, scale=FLAGS.image_scale, training=False)
	    elif FLAGS.model_type == "alipay":
	      generated = al_model.net(rawimage, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            start_time = time.time()
            generated = sess.run(generated)
            print(generated.shape)
            end_time = time.time()
            tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
	    if FLAGS.same_shape:
		generated = tf.image.resize_images(generated, [height, width])
            generated = tf.cast(generated, tf.uint8)
            generated_file = 'generated/aares_%s.jpg' % (FLAGS.model_type)
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
		tf.logging.info('generated Image size: %s' % (generated.get_shape()))
                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
