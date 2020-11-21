import tensorflow as tf
from layers import fc_layer, fc_layer_linear


class Camnet(object):
    def __init__(self, latent_dim, output_dim, name_scope = 'Camnet'):
        self.latent_dim = latent_dim
        self.name_scope = name_scope
        self.output_dim = output_dim

    def forward(self, input_feature):
        with tf.variable_scope(self.name_scope):

            lc11 = fc_layer(input_feature, self.latent_dim, name='camfc1')
            lc12 = fc_layer(lc11, self.latent_dim, name='camfc2')
            lc13 = input_feature + lc12

            lc21 = fc_layer(lc13, self.latent_dim, name='camfc3')
            lc22 = fc_layer(lc21, self.latent_dim, name='camfc4')
            lc23 = lc13 + lc22

            lc_out = fc_layer_linear(lc23, self.output_dim, name='camfc5')

            return lc_out


class Posenet(object):
    def __init__(self, latent_dim, output_dim, name_scope = 'Posenet'):

        self.latent_dim = latent_dim
        self.name_scope = name_scope
        self.output_dim = output_dim

    def forward(self, input_feature, noise):
        with tf.variable_scope(self.name_scope):
            concat = tf.concat([input_feature, noise], axis=-1)
            lp11 = fc_layer(concat, self.latent_dim, name='posefc1')
            lp12 = fc_layer(lp11, self.latent_dim, name='posefc2')
            lp13 = input_feature + lp12

            lp21 = fc_layer(lp13, self.latent_dim, name='posefc3')
            lp22 = fc_layer(lp21, self.latent_dim, name='posefc4')
            lp23 = lp13 + lp22
            lc_out = fc_layer_linear(lp23, self.output_dim, name='posefc5')

            return lc_out


class Encoder(object):
    def __init__(self, latent_dim, output_dim, name_scope = 'Encoder'):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.name_scope = name_scope

    def forward(self, input_pose):
        with tf.variable_scope(self.name_scope):
            le10 = fc_layer(input_pose, self.latent_dim, name='encfc1')
            le11 = fc_layer(le10, self.latent_dim, name='encfc2')
            le12 = fc_layer(le11, self.latent_dim, name='encfc3')
            le13 = le10 + le12
            le_out = fc_layer_linear(le13, self.output_dim, name='encfc4')
            return le_out


class Posenet_repnet(object):
    def __init__(self, latent_dim, output_dim, name_scope = 'Posenet'):

        self.latent_dim = latent_dim
        self.name_scope = name_scope
        self.output_dim = output_dim

    def forward(self, input_feature):
        with tf.variable_scope(self.name_scope):
            lp11 = fc_layer(input_feature, self.latent_dim, name='posefc1')
            lp12 = fc_layer(lp11, self.latent_dim, name='posefc2')
            lp13 = input_feature + lp12

            lp21 = fc_layer(lp13, self.latent_dim, name='posefc3')
            lp22 = fc_layer(lp21, self.latent_dim, name='posefc4')
            lp23 = lp13 + lp22
            lc_out = fc_layer_linear(lp23, self.output_dim, name='posefc5')

            return lc_out


class discriminator(object):
    def __init__(self, latent_dim_pose, latent_dim_kcs, output_dim, name_scope = 'Discriminator'):
        self.latent_dim_pose = latent_dim_pose
        self.latent_dim_kcs = latent_dim_kcs
        self.output_dim = output_dim
        self.name_scope = name_scope

    def forward(self, pose, kcs, reuse = False):
        with tf.variable_scope(self.name_scope) as vs:
            if (reuse):
                vs.reuse_variables()
            ldp10 = fc_layer(pose, self.latent_dim_pose, name='discfc0')
            ldp11 = fc_layer(ldp10, self.latent_dim_pose, name='discfc1')
            ldp12 = fc_layer(ldp11, self.latent_dim_pose, name='discfc2')
            ldp13 = ldp10 + ldp12
            ldp21 = fc_layer(ldp13, self.latent_dim_pose, name='discfc3')
            ldp22 = fc_layer(ldp21, self.latent_dim_pose, name='discfc4')
            ldp23 = ldp13 + ldp22

            ldk10 = fc_layer(kcs, self.latent_dim_kcs, name='discfc5')
            ldk11 = fc_layer(ldk10, self.latent_dim_kcs, name='discfc6')
            ldk12 = fc_layer(ldk11, self.latent_dim_kcs, name='discfc7')
            ldk13 = ldk10 + ldk12
            ldk21 = fc_layer(ldk13, self.latent_dim_kcs, name='discfc8')
            ldk22 = fc_layer(ldk21, self.latent_dim_kcs, name='discfc9')
            ldk23 = ldk13 + ldk22

            ld = tf.concat([ldp23, ldk23], axis=-1)
            ld_out = fc_layer_linear(ld, self.output_dim, name='dicsfc10')

            return ld_out
