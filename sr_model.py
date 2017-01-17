import tensorflow as tf
from layer import conv2d, resize_conv2d, instance_norm, residual_block

slim = tf.contrib.slim
debug = False

def net(image, scale, training):
  with tf.variable_scope('sr_model'):
   with slim.arg_scope([slim.conv2d], 
           weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
           activation_fn=None,
           padding="VALID"):

    with tf.variable_scope('conv1'):
    	net = tf.nn.relu(instance_norm(conv2d(image, 64, 5, 1)))
    if debug: print("conv1", net.get_shape())
    with tf.variable_scope('conv2'):
    	net = tf.nn.relu(instance_norm(conv2d(net, 128, 3, 2)))
    if debug: print("conv2",net.get_shape())
    #with tf.variable_scope('conv3'):
    #	net = tf.nn.relu(instance_norm(conv2d(net, 128, 3, 2)))
    #if debug: print("conv3",net.get_shape())
    net = slim.repeat(net, 5, residual_block, filters=128, scope="residual")
    i = 1
    while scale / 2 != 0:
      scale /= 2
      with tf.variable_scope('deconv%s'%i):
          net = tf.nn.relu(instance_norm(resize_conv2d(net, 128 / (2**i), 3, 2, training)))
      if debug: print("dconv",i,net.get_shape())
      i+=1
    with tf.variable_scope('deconv%s'%i):
        net = tf.nn.relu(instance_norm(resize_conv2d(net, 128 / (2**i), 3, 2, training)))
    if debug: print("dconv",i,net.get_shape())
    with tf.variable_scope('fconv3'):
        net = tf.nn.tanh(instance_norm(conv2d(net, 3, 5, 1)))
    if debug: print("fconv3",net.get_shape())

    net = (net + 1) * 127.5
    return net
