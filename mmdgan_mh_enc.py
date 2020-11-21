import tensorflow as tf
import tensorflow.contrib.layers as tcl
from layers import fc_layer, resisual_block, kcs_layer, reprojection_layer
import ops
import os
import mmd


class Pose_mmdgan_enc(object):
    def __init__(self, posenet, camnet, critic, encoder, latent_dim, batchsize, summary_dir, epochs, pose2d_dim, pose3d_dim, kernel,
                 reproweight = 100.0, cam_weight = 100.0, gp_weight = 1.0, reg_weight=10.0, dot_weight = 10.0, enc_weight = 10.0,
                 beta1 = 0.5,
                 dtype = tf.float32):
        self.posenet = posenet
        self.camnet = camnet
        self.critic = critic
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.inputsize = pose2d_dim
        self.outputsize = pose3d_dim
        self.kernel = kernel
        self.dtype = dtype
        self.batchsize = batchsize
        self.summary_dir = summary_dir
        self.epochs = epochs
        self.training_ratio = 5
        self.repro_weight = reproweight
        self.cam_weight = cam_weight
        self.gp_weight = gp_weight
        self.reg_weight = reg_weight
        self.enc_weight = enc_weight
        self.beta1 = beta1
        self.dot_weight = dot_weight
        self.train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'))
        self.test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'))
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build_model(self):

        with tf.variable_scope('Input'):
            generator_in = tf.placeholder(self.dtype, shape=[self.batchsize, self.inputsize], name='generator_in')
            generator_in_noise = tf.placeholder(self.dtype, shape=[self.batchsize, self.latent_dim], name='generator_in_noise')
            discriminator_in = tf.placeholder(self.dtype, shape=[self.batchsize, self.outputsize], name='discriminator_in')

        self.generator_in = generator_in
        self.discriminator_in = discriminator_in
        self.generator_in_noise = generator_in_noise
        self.isTraining = tf.placeholder(tf.bool, name='isTrainingflag')
        self.lr_d = tf.placeholder(self.dtype, name='learning_rate_d')
        self.lr_g = tf.placeholder(self.dtype, name='learning_rate_g')

        with tf.variable_scope('Generator') :
            h1 = fc_layer(generator_in, self.latent_dim, name='genfc1')
            h2 = resisual_block(h1, self.latent_dim, name_scope='Block1')
            pose_out = self.posenet.forward(h2, generator_in_noise)
            cam_out = self.camnet.forward(h2)
            enc_out = self.encoder.forward(pose_out)

        psi_out = kcs_layer(pose_out)
        psi_vec_out = tcl.flatten(psi_out)

        psi_real = kcs_layer(discriminator_in)
        psi_vec_real = tcl.flatten(psi_real)

        average_samples = ops.weightedsample(discriminator_in, pose_out)
        psi_average_samples = kcs_layer(average_samples)
        psi_average_samples_vec = tcl.flatten(psi_average_samples)

        pose2d_repro = reprojection_layer(cam_out, pose_out)

        with tf.variable_scope('Discriminator') :
            d_fake = self.critic.forward(pose_out, psi_vec_out)
            d_real = self.critic.forward(discriminator_in, psi_vec_real, reuse = True)
            d_average = self.critic.forward(average_samples, psi_average_samples_vec, reuse = True)
        self.d_real = d_real
        self.d_fake = d_fake

        with tf.variable_scope('loss') :
            self.loss_reprojection = ops.weighted_pose_2d_loss(generator_in, pose2d_repro)
            self.loss_cam = ops.cam_loss(cam_out)
            self.loss_reg = ops.regularizer(pose_out, generator_in_noise)

            kernel = getattr(mmd, '_%s_kernel' % self.kernel)
            K_XX, K_XY, K_YY, wts = kernel(d_fake, d_real, add_dot = self.dot_weight)
            self.loss_g = mmd.mmd2([K_XX, K_XY, K_YY, False])
            self.loss_d = - self.loss_g
            self.loss_gp = mmd.gp_loss(d_average, average_samples, d_real, d_fake, kernel, self.dot_weight)
            self.loss_enc = tf.reduce_mean(tf.abs(enc_out - self.generator_in_noise))
        total_loss_g = self.repro_weight * self.loss_reprojection + self.cam_weight * self.loss_cam  + \
                       self.loss_g + self.reg_weight * self.loss_reg + self.enc_weight * self.loss_enc
        total_loss_d = self.gp_weight * self.loss_gp + self.loss_d

        self.output_pose = pose_out
        self.camout = cam_out

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if  'Discriminator' in var.name]
        self.g_var = [var for var in t_vars if 'Generator' in var.name]
        self.d_optimim = tf.train.AdamOptimizer(self.lr_d, beta1=self.beta1, beta2=0.9)
        self.g_optimim = tf.train.AdamOptimizer(self.lr_g, beta1=self.beta1, beta2=0.9)
        dgradients = self.d_optimim.compute_gradients(total_loss_d, self.d_vars)
        ggradients = self.g_optimim.compute_gradients(total_loss_g, self.g_var)
        # dgradients = [(tf.clip_by_norm(dd, 1.0), vv) for dd, vv in dgradients]
        # ggradients = [(tf.clip_by_norm(gg, 1.0), vv) for gg, vv in ggradients]
        self.dupdates = self.d_optimim.apply_gradients(dgradients, global_step=self.global_step)
        self.gupdates = self.g_optimim.apply_gradients(ggradients, global_step=self.global_step)

        self.loss_repro_summ = tf.summary.scalar('loss/loss_repro', self.loss_reprojection, collections=['train', 'test'])
        self.loss_cam_summ = tf.summary.scalar('loss/loss_cam', self.loss_cam, collections=['train', 'test'])
        self.loss_gp_summ = tf.summary.scalar('loss/loss_gp', self.loss_gp, collections=['train'])
        self.loss_d_summ = tf.summary.scalar('loss/loss_d', self.loss_d, collections=['train'])
        self.loss_g_summ = tf.summary.scalar('loss/loss_g', self.loss_g, collections=['train'])
        self.learning_rate_summary_d = tf.summary.scalar('learning_rate_d', self.lr_d)
        self.learning_rate_summary_g = tf.summary.scalar('learning_rate_g', self.lr_g)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def DStep(self, sess, poses2d, poses3d, noise, lr_d, isTraining):
        _, loss_d, loss_gp, loss_d_summ, loss_g_summ, lr_summ = sess.run([self.dupdates,
                                                                 self.loss_d,
                                                                 self.loss_gp,
                                                                 self.loss_d_summ,
                                                                 self.loss_gp_summ,
                                                                 self.learning_rate_summary_d],
                                                      feed_dict = {self.generator_in: poses2d,
                                                                   self.discriminator_in: poses3d,
                                                                   self.generator_in_noise: noise,
                                                                   self.lr_d: lr_d,
                                                                   self.isTraining: isTraining})
        # self.train_writer.add_summary(loss_d_summ, self.global_step)
        # self.train_writer.add_summary(loss_g_summ, self.global_step)
        # self.train_writer.add_summary(lr_summ, self.global_step)
        return loss_d, loss_gp

    def Gstep(self, sess, poses2d, poses3d, noise, lr_g, isTraining):
        _, loss_g, loss_repro, loss_cam, loss_reg, loss_enc, loss_g_summ, loss_repro_summ, loss_cam_summ = sess.run([  self.gupdates,
                                                                                                   self.loss_g,
                                                                                                   self.loss_reprojection,
                                                                                                   self.loss_cam,
                                                                                                   self.loss_reg,
                                                                                                   self.loss_enc,
                                                                                                   self.loss_g_summ,
                                                                                                   self.loss_repro_summ,
                                                                                                   self.loss_cam_summ],
                                                                                                  feed_dict = {self.generator_in: poses2d,
                                                                                                               self.discriminator_in: poses3d,
                                                                                                               self.generator_in_noise: noise,
                                                                                                               self.lr_g: lr_g,
                                                                                                               self.isTraining: isTraining})

        return loss_g, loss_repro, loss_cam, loss_reg, loss_enc

    def inference(self, sess, poses2d, poses3d, noise, lr_g):
        pose_out, cam_out, d_real, d_fake = sess.run([self.output_pose, self.camout, self.d_real, self.d_fake],
                            feed_dict = {self.generator_in: poses2d,
                                         self.discriminator_in: poses3d,
                                         self.generator_in_noise: noise,
                                         self.lr_g: lr_g,
                                         self.isTraining: False})

        return pose_out, cam_out




