import tensorflow as tf
from layer import conv2d, resize_conv2d, instance_norm, residual_block
slim = tf.contrib.slim

debug = False

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
