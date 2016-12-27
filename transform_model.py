import tensorflow as tf

slim = tf.contrib.slim
debug = False

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
      if debug: print("residual",net.get_shape())
      return net


def net(image, training):
    # Less border effects when padding a little before passing through ..
  image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

  with slim.arg_scope([slim.conv2d], 
           weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
           activation_fn=None,
           padding="VALID"):

    with tf.variable_scope('conv1'):
    	net = tf.nn.relu(instance_norm(conv2d(image, 32, 9, 1)))
    if debug: print("conv1", net.get_shape())
    with tf.variable_scope('conv2'):
    	net = tf.nn.relu(instance_norm(conv2d(net, 64, 3, 2)))
    if debug: print("conv2",net.get_shape())
    with tf.variable_scope('conv3'):
    	net = tf.nn.relu(instance_norm(conv2d(net, 128, 3, 2)))
    if debug: print("conv3",net.get_shape())
    net = slim.repeat(net, 5, residual_block, scope="residual")
    with tf.variable_scope('deconv1'):
        # deconv1 = tf.nn.relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
        net = tf.nn.relu(instance_norm(resize_conv2d(net, 64, 3, 2, training)))
    if debug: print("dconv1",net.get_shape())
    with tf.variable_scope('deconv2'):
        # deconv2 = tf.nn.relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
        net = tf.nn.relu(instance_norm(resize_conv2d(net, 32, 3, 2, training)))
    if debug: print("dconv2",net.get_shape())
    with tf.variable_scope('deconv3'):
        # deconv_test = tf.nn.relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1)))
        net = tf.nn.tanh(instance_norm(conv2d(net, 3, 9, 1)))
    if debug: print("dconv3",net.get_shape())

    net = (net + 1) * 127.5
    # Remove border effect reducing padding.
    height = tf.shape(net)[1]
    width = tf.shape(net)[2]
    net = tf.slice(net, [0, 10, 10, 0], tf.pack([-1, height - 20, width - 20, -1]))

    return net
