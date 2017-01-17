import tensorflow as tf
import numpy as np
import yaml
from PIL import Image
from os import listdir
from os.path import isfile, join

slim = tf.contrib.slim


def get_image(filepath, height, width, preprocess_fn, queue=None):
    png = filepath.lower().endswith('png')
    if queue is None:
      img_bytes = tf.read_file(filepath)
    else:
      reader = tf.WholeFileReader()
      _, img_bytes = reader.read(queue)

    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)


def batch_image(FLAGS, preprocess_fn):

    def st_batch_image(FLAGS, preprocess_fn):
        filenames = [join(FLAGS.train_dir, filename) for filename in listdir(FLAGS.train_dir) \
                          if isfile(join(FLAGS.train_dir, filename))]
        images = train_batch(filenames, FLAGS, preprocess_fn)
        return images, images
    
    
    def sr_batch_image(FLAGS, preprocess_fn):
        filenames = [join(FLAGS.train_dir, filename) for filename in listdir(FLAGS.train_dir) \
                          if isfile(join(FLAGS.train_dir, filename))][:FLAGS.train_dataset_size]
        super_images = train_batch(filenames, FLAGS, preprocess_fn)
        gaussian_filter = get_gaussian_filter(1.0, 5)
        smoothed_images = tf.nn.depthwise_conv2d(super_images, gaussian_filter, [1,1,1,1], padding="SAME")
        low_images = tf.image.resize_bicubic(smoothed_images, [FLAGS.image_size/FLAGS.image_scale, FLAGS.image_size/FLAGS.image_scale])
        tf.logging.info('make super resolution dataset %s' % len(filenames))
        return low_images, super_images

    def al_batch_image(FLAGS, preprocess_fn):
        filenames = [join(FLAGS.train_dir, filename) for filename in listdir(FLAGS.train_dir) \
                          if isfile(join(FLAGS.train_dir, filename))]
        origin_images = train_batch(filenames, FLAGS, preprocess_fn)
        if FLAGS.network == "color_line":
          masked_images = get_masks(origin_images, FLAGS.image_size, FLAGS.image_size)
        else:
          #gaussian_filter = get_gaussian_filter(1.0, 5)
          #smoothed_images = tf.nn.depthwise_conv2d(origin_images, gaussian_filter, [1,1,1,1], padding="SAME")
          masked_images = get_mask_file(origin_images, FLAGS.mask_file, FLAGS.image_size, FLAGS.image_size)
        tf.logging.info('make masked dataset %s' % len(filenames))
        return masked_images, origin_images

    network_batch_fn = {"style": st_batch_image, "super":sr_batch_image,"color_line":al_batch_image, "alipay": al_batch_image}

    return network_batch_fn[FLAGS.network](FLAGS, preprocess_fn)


def train_batch(filenames, FLAGS, preprocess_fn):
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=FLAGS.epoch)
    processed_image = get_image(filenames[0], FLAGS.image_size, FLAGS.image_size, preprocess_fn, queue=filename_queue)
    return tf.train.batch([processed_image], FLAGS.batch_size, dynamic_pad=True)

def get_gaussian_filter(sigma, size, channels=3):
    m = (size - 1.) / 2.
    y, x = np.ogrid[-m: m + 1, -m: m + 1]
    y = np.array(y, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    density = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    density[density < np.finfo(density.dtype).eps * density.max()] = 0
    density /= density.sum()
    ft = density.reshape((size, size, 1, 1))
    gf = np.repeat(ft, channels, axis = 2)
    return tf.convert_to_tensor(gf)

def get_mask_file(origin_images, mask_file, height, width, channels=3):
    """blur image through a mask file"""
    img_bytes = tf.read_file(mask_file)
    maskimage = tf.image.decode_jpeg(img_bytes)
    maskimage = tf.to_float(maskimage)

    m_mean = tf.reduce_mean(maskimage, axis=(1,2))
    index = tf.where(m_mean < 1.5)
    side_index = tf.where(m_mean >= 1.5)
    top_index = side_index + tf.to_int64(1)
    down_index = side_index - tf.to_int64(1)

    select = tf.zeros_like(m_mean, dtype=tf.float32)
    side_select = tf.ones_like(m_mean, dtype=tf.float32)
    values = tf.squeeze(tf.ones_like(index, dtype=tf.float32))
    side_values = tf.squeeze(tf.ones_like(side_index, dtype=tf.float32))
    top_values = tf.scalar_mul(tf.random_uniform([], minval=0, maxval=1), side_values)
    down_values = tf.scalar_mul(tf.random_uniform([], minval=0, maxval=1), side_values)

    delta = tf.SparseTensor(index, values, [height])
    top_delta = tf.SparseTensor(top_index, top_values, [height])
    down_delta = tf.SparseTensor(down_index, down_values, [height])

    black_select = select + tf.sparse_tensor_to_dense(delta)
    top_select = side_select + tf.sparse_tensor_to_dense(top_delta)
    down_select = side_select + tf.sparse_tensor_to_dense(down_delta)

    top_select = tf.expand_dims(tf.divide(tf.ones_like(top_select), top_select), -1)
    top_select = tf.matmul(top_select, tf.ones([1, width]))
    top_select = tf.expand_dims(top_select, -1)
    down_select = tf.expand_dims(tf.divide(tf.ones_like(down_select), down_select), -1)
    down_select = tf.matmul(down_select, tf.ones([1, width]))
    down_select = tf.expand_dims(down_select, -1)

    black_select = tf.expand_dims(black_select, -1)
    black_select = tf.matmul(black_select, tf.ones([1, width]))
    black_select = tf.expand_dims(black_select, 0)
    black_select = tf.expand_dims(black_select, -1)
    top_select = tf.expand_dims(top_select, 0)
    down_select = tf.expand_dims(down_select, 0)

    source = tf.mul(origin_images, top_select)
    source = tf.mul(source, down_select)
    source = tf.mul(source, black_select)

    return source

def get_masks(origin_images, height, width, channels=3):
    """add horizon color lines and set empty"""
    quarty = tf.random_uniform([height/4, 1])
    prop = tf.scalar_mul(tf.convert_to_tensor(0.2), tf.ones([height/4, 1]))
    quarty = tf.round(tf.add(quarty, prop))
    y = tf.reshape(tf.stack([quarty, quarty, quarty, quarty], axis=1), [height, 1])
    mask = tf.matmul(y, tf.ones([1, width]))
    masks = tf.expand_dims(mask, 0)
    masks = tf.expand_dims(masks, -1)
    maskedimages = tf.mul(origin_images, masks)
    """add noise"""
    scale = tf.random_uniform([channels, height, 1])
    y = tf.subtract(tf.ones([height, 1]), y)
    y = tf.expand_dims(y, 0)
    y = tf.scalar_mul(tf.convert_to_tensor(255.), tf.multiply(scale, y))
    noise = tf.add(mask, tf.matmul(y, tf.ones([channels, 1, width])))
    noise = tf.pack(tf.split(value=noise, num_or_size_splits=noise.get_shape()[0], axis=0), axis=3)
    maskedimages = tf.add(maskedimages, noise)
    return maskedimages

def _get_init_fn(FLAGS):
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


def read_conf_file(conf_file):
    class Flag(object):
      def __init__(self, content):
        self.__dict__ = dict(content)
    with open(conf_file) as f:
        FLAGS = Flag(yaml.load(f))
    return FLAGS
