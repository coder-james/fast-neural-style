#!/usr/bin/python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from preprocessing.vgg_preprocessing import unprocess_image
import transform_model,sr_model, al_model
import time
import losses
import utils
import os,sys
import argparse

slim = tf.contrib.slim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', default='alipay', help="model network('style','super','color_line','alipay')")
    parser.add_argument('-c', '--conf', default='conf/alipay.yml', help='the path to the conf file')
    return parser.parse_args()

def main(FLAGS):
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    if FLAGS.network == "style":
      """precompute style feature"""
      style_features_t = losses.get_style_features(FLAGS)

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            """Build Network"""
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model, 1)

            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)

            input_images, content_images = utils.batch_image(FLAGS, image_preprocessing_fn)
            tf.logging.info('Network Input Images size %s' % input_images.get_shape())
            tf.logging.info('Content Images size %s' % content_images.get_shape())

            if FLAGS.network == "style":
              generated = transform_model.net(input_images, training=False)
            elif FLAGS.network == "super":
              generated = sr_model.net(input_images, scale=FLAGS.image_scale, training=False)
	    elif FLAGS.network == "alipay" or FLAGS.network == "color_line":
	      generated = al_model.net(input_images, training=False)
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unpack(generated, axis=0, num=FLAGS.batch_size)
                                   ]
            
            processed_generated = tf.pack(processed_generated)
            concat_input = tf.concat(0, [processed_generated, content_images])
            _, endpoints_dict = network_fn(concat_input)
            for key in endpoints_dict:
                tf.logging.info(key)

            tf.summary.scalar('batch_size', FLAGS.batch_size)
             
            content_loss, content_loss_summary = losses.content_loss(endpoints_dict, FLAGS.content_layers, FLAGS.content_weights)
            tf.summary.scalar('losses/content_loss', content_loss)
            for layer in FLAGS.content_layers:
              tf.summary.scalar('losses/' + layer, content_loss_summary[layer])
            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tv_loss = losses.total_variation_loss(generated) 
            #tf.summary.scalar('losses/regularizer_loss', tv_loss)
            #tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * FLAGS.tv_weight)

            if FLAGS.network == "style":
              style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
              tf.summary.scalar('losses/style_loss', style_loss)
              tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * FLAGS.style_weight)
              loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss
              for layer in FLAGS.style_layers:
                  tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            elif FLAGS.network == "alipay":
              pixel_loss = losses.pixel_loss(concat_input, FLAGS)
              tf.summary.scalar('losses/pixel_loss', pixel_loss)
              tf.summary.scalar('weighted_losses/weighted_pixel_loss', pixel_loss * FLAGS.pixel_weight)
	      loss = FLAGS.content_weight * content_loss + FLAGS.pixel_weight * pixel_loss + FLAGS.tv_weight * tv_loss
            else:
	      loss = FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

            tf.summary.scalar('total_loss', loss)
            tf.summary.image('generated', generated)
            tf.summary.image('input', tf.pack([
                unprocess_image(image) for image in tf.unpack(input_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)
            variable_to_train = []
            for variable in tf.trainable_variables():
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)

            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            """Start Training"""
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-%s-model.ckpt' % FLAGS.network), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-%s-model.ckpt-done' % FLAGS.network))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    FLAGS.network = args.network
    main(FLAGS)
