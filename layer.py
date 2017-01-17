import tensorflow as tf
slim = tf.contrib.slim

def conv2d(net, output_filters, kernel, stride, mode='REFLECT', relu=False):
    net = tf.pad(net, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
    if relu:
      return slim.conv2d(net, output_filters, kernel, stride=stride, activation_fn=tf.nn.relu)
    else:
      return slim.conv2d(net, output_filters, kernel, stride=stride)


def resize_conv2d(net, output_filters, kernel, stride, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    height = net.get_shape()[1].value if training else tf.shape(net)[1]
    width = net.get_shape()[2].value if training else tf.shape(net)[2]

    new_height = height * stride * 2
    new_width = width * stride * 2

    net = tf.image.resize_images(net, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return conv2d(net, output_filters, kernel, stride)


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(var, epsilon)))


def residual_block(net, filters=128, kernel=3, stride=1, scope=None):
    with tf.variable_scope(scope, 'residual'):
      tower1 = conv2d(net, filters, kernel, stride, relu=True)
      tower2 = conv2d(tower1, filters, kernel, stride)
      net = net + tower2
      return net
