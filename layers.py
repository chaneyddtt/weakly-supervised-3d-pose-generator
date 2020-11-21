import tensorflow as tf


def weight_variable(name, shape):

    initer = tf.contrib.layers.variance_scaling_initializer()
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):

    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def fc_layer_linear(x, num_units, name):

    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    return layer


def fc_layer(x, num_units, name):

    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    layer = tf.nn.leaky_relu(layer)
    return layer


def reprojection_layer(cam, pose3d):

    cam = tf.reshape(tf.to_float(cam), [-1, 2, 3])
    pose3d = tf.reshape(tf.to_float(pose3d), [-1, 3, 16])
    pose2d = tf.reshape(tf.matmul(cam, pose3d), [-1, 32])
    return pose2d


def kcs_layer(pose3d):
    # modify the last two columns of the orginal matrix left hip->hip, right hip->hip
    Ct = tf.constant([
          [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 1, 0],
          [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 , 0, 0, 0, -1,-1],
          [0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,-1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,-1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,-1, 0, 0]])

    C = tf.reshape(tf.tile(Ct, (tf.shape(pose3d)[0], 1)), (-1, 16, 15))

    poses3 = tf.to_float(tf.reshape(pose3d, [-1, 3, 16]))
    B = tf.matmul(poses3, C)
    Psi = tf.matmul(tf.transpose(B, perm=[0, 2, 1]), B)

    return Psi


def resisual_block(input, latent_dim, name_scope):
    with tf.name_scope(name_scope):
        l1 = fc_layer(input, latent_dim, name='fc1')
        l2 = fc_layer(l1, latent_dim, name='fc2')
        out = input + l2

    return out