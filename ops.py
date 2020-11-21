import tensorflow as tf
import numpy as np


def weightedsample(realsample, fakesample):
    alpha = tf.random.uniform(shape=[tf.shape(realsample)[0], 1])
    return alpha * realsample + (1.0 - alpha) * fakesample


def weighted_pose_2d_loss(pose2d_gt, pose2d_repro):

    diff = tf.to_float(tf.abs(pose2d_gt - pose2d_repro))

    # weighting the joints
    weights_t = tf.to_float(
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))

    weights = tf.tile(tf.reshape(weights_t, (1, 32)), (tf.shape(pose2d_repro)[0], 1))

    tmp = tf.multiply(weights, diff)

    loss = tf.reduce_sum(tmp, axis=1) / 32

    return tf.reduce_mean(loss)


def weighted_pose_2d_loss_mj(pose2d_gt, pose2d_repro, missing_idx):
    diff = tf.to_float(tf.abs(pose2d_gt - pose2d_repro))

    # weighting the joints
    weights_t = tf.to_float(
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))
    weights = tf.tile(tf.reshape(weights_t, (1, 32)), (tf.shape(pose2d_repro)[0], 1))
    weights_f = tf.multiply(missing_idx, weights)
    tmp = tf.multiply(weights_f, diff)
    loss = tf.reduce_sum(tmp, axis=1) / 32

    return tf.reduce_mean(loss)


def wassertein_loss(d_real, d_fake):
    loss = tf.reduce_mean(d_fake - d_real)
    return loss


def cam_loss(cam_param):
    m = tf.reshape(cam_param, [-1, 2, 3])

    m_sq = tf.matmul(m, tf.transpose(m, perm=[0, 2, 1]))

    loss_mat = tf.reshape((2 / tf.trace(m_sq)), [-1, 1, 1])*m_sq - tf.eye(2)

    loss = tf.reduce_sum(tf.abs(loss_mat), axis=[1, 2])

    return tf.reduce_mean(loss)


def gp_loss(d_out, average_samples):

    gradients = tf.gradients(d_out, [average_samples])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return gradient_penalty


def regularizer(g_out, z):
    batch_size_h = tf.to_int32(tf.shape(g_out)[0]/2)
    pose1 = tf.slice(g_out, [0, 0], [batch_size_h, -1])
    pose2 = tf.slice(g_out, [batch_size_h, 0], [batch_size_h, -1])
    z1 = tf.slice(z, [0, 0], [batch_size_h, -1])
    z2 = tf.slice(z, [batch_size_h, 0], [batch_size_h, -1])
    pose_diff = tf.reduce_mean(tf.abs(pose1 - pose2), axis = 1)
    z_diff = tf.reduce_mean(tf.abs(z1 - z2), axis =1)
    loss = tf.nn.relu(1.0 - pose_diff / z_diff)

    return tf.reduce_mean(loss)


def compute_R(K):
    s = np.linalg.norm(K, ord=2, axis=(1, 2), keepdims=True)
    r = K/s
    # s = np.sqrt(np.trace(np.matmul(K, np.transpose(K, [0, 2, 1])), axis1=1, axis2=2)/2)
    # s = s[:, np.newaxis, np.newaxis]
    # r = K/s
    r_x = r[:, 0:1, :]
    r_y = r[:, 1:2, :]
    r_z = np.cross(r_x, r_y)
    R = np.zeros([r.shape[0], 3, 3])
    for i in range(r.shape[0]):
        R[i, 0:2, :] = r[i, :, :]
        R[i, 2, :] = r_z[i, :, :]
    return R
