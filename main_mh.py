import numpy as np
import argparse
from base_module import Posenet, Camnet, discriminator, Encoder
from mmdgan_mh_enc import Pose_mmdgan_enc
import os
import random
import tensorflow as tf
import scipy.io as sio
import logging, logging.config
import sys
from eval_functions import err_3dpe

parse = argparse.ArgumentParser()
parse.add_argument("--batchsize", help= "the batch size used in training", default=128, type=int)
parse.add_argument("--epochs", help="number of epochs for training", default=50, type=int)
parse.add_argument("--latent_dim", help="dimension of latent space", default=1024, type=int)
parse.add_argument("--latent_dim_pose", help="dimension for pose in the latent space of discriminator", default=128, type=int)
parse.add_argument("--latent_dim_kcs", help="dimension for kcs in the latent space of discriminator", default=1024, type=int)
parse.add_argument("--d_output_dim", help="dimension for output of discriminator", default=8, type=int)
parse.add_argument("--lr", help="learning rate", default=1e-4, type=float)
parse.add_argument("--architecture", help="which architeture to use[mmdgan, mmdgan_enc]", default='mmdgan_enc', type=str)
parse.add_argument("--beta1", help="beta1 for adamoptimizor", default=0.5, type=float)
parse.add_argument("--diter", help="the number of discriminator updates oer generator updates", default=1, type=int)
parse.add_argument("--kernel", help="kernel type used in mmd[dot, mix_rbf, mix_rq]", default='mix_rq', type=str)
parse.add_argument("--repro_weight", help="weight of reprojection loss", default=10.0, type=float)
parse.add_argument("--cam_weight", help="weight of camera loss", default=10.0, type=float)
parse.add_argument("--gp_weight", help="weight of dot kernel in mix kernel", default=0.1, type=float)
parse.add_argument("--reg_weight", help="weight for regularizer", default=7.5, type=float)
parse.add_argument("--dot_weight", help="weight of dot kernel in mix kernel", default=10.0, type=float)
parse.add_argument("--lr_decay", help="learning rate decay rate", default=0.94, type=float)

parse.add_argument("--enc_weight", help="weight of encoder", default=10.0, type=float)
parse.add_argument("--checkpoint", help="which model to load", default=0, type=int)
parse.add_argument("--num_samples", help="number of hypotheses", default=10, type=int)
parse.add_argument("--datatype", help="datatype used for training [GT, SHFT, GTMJ8-1, GTMJ8-2]", default='GT', type=str)

args = parse.parse_args()
print(args)

pose3d_dim = 16 * 3
pose2d_dim = 16 * 2
cam_dim = 6
lr = args.lr
model_name = '{}_regweight{}_encweight{}_2D{}_test'.format(args.architecture, args.reg_weight, args.enc_weight, args.datatype)
log_dir = os.path.join('logs_test', model_name)
models_dir = os.path.join(log_dir, 'models')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logging.config.fileConfig('./logging.conf')
logger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/log.txt".format(log_dir))
# logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
# fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
logger.info("Logs will be written to %s" % log_dir)
def log_arguments():
    logger.info('Command: %s', ' '.join(sys.argv))
    s = '\n'.join(['    {}: {}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    s = 'Arguments:\n' + s
    logger.info(s)


log_arguments()
posenet = Posenet(args.latent_dim, pose3d_dim)
camnet = Camnet(args.latent_dim, cam_dim)
disc = discriminator(args.latent_dim_pose, args.latent_dim_kcs, args.d_output_dim)

encoder = Encoder(args.latent_dim, args.latent_dim)
repnet = Pose_mmdgan_enc(posenet, camnet, disc, encoder, args.latent_dim, args.batchsize, log_dir, args.epochs, pose2d_dim, pose3d_dim,
                 args.kernel, args.repro_weight, args.cam_weight, args.gp_weight, args.reg_weight, args.dot_weight, args.enc_weight)
repnet.build_model()

poses = sio.loadmat('new_data/data_2d{}_3d_train.mat'.format(args.datatype))
poses_3d = poses['poses_3d']/1000
poses_2d = poses['poses_2d']

# randomly permute training data
rp = np.random.permutation(poses_3d.shape[0])
poses3d = poses_3d[rp, :]
rp = np.random.permutation(poses_2d.shape[0])
poses2d = poses_2d[rp, :]

poses_eval = sio.loadmat('new_data/data_2d{}_3d_test.mat'.format(args.datatype))
poses_2d_eval = poses_eval['poses_2d']
poses_3d_eval = poses_eval['poses_3d']/1000
rp = np.random.permutation(poses_3d_eval.shape[0])
poses2d_eval = poses_2d_eval[rp, :]
poses3d_eval = poses_3d_eval[rp, :]
poses2d_eval = poses2d_eval[:1000, :]
poses3d_eval = poses3d_eval[:1000, :]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
best_val = 100.0
with tf.Session(config=config) as sess:
    batchsize = args.batchsize
    sess.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        np.random.shuffle(poses3d)
        batch_size_half = np.int32(batchsize/2)
        minibatch_size = batch_size_half * repnet.training_ratio
        logger.info('Epoch:{}'.format(epoch))
        logger.info('Number of batches: {}'.format(int(poses2d.shape[0] // batch_size_half)))
        discriminator_losses = []
        adversarial_losses = []

        for i in range(poses2d.shape[0] // minibatch_size):
            poses2d_minibatch = poses2d[i * minibatch_size: (i + 1) * minibatch_size]
            random_samples = random.sample(range(0, poses3d.shape[0]), minibatch_size)
            discriminator_minibatch = poses3d[random_samples]
            for j in range(repnet.training_ratio):
                poses3d_batch_half = discriminator_minibatch[j * batch_size_half: (j + 1) * batch_size_half]
                poses3d_batch = np.concatenate([poses3d_batch_half, poses3d_batch_half], axis=0)
                poses2d_batch_half = poses2d_minibatch[j * batch_size_half: (j + 1) * batch_size_half]
                # only use generate half batch size because we need to generate a pair of 3d poses for each 2d pose
                # such that we can add the regularizer
                poses2d_batch = np.concatenate([poses2d_batch_half, poses2d_batch_half], axis=0)
                noise = np.random.normal(0, 1, (batchsize, args.latent_dim))
                for k in range(args.diter):
                    loss_d, loss_gp = repnet.DStep(sess, poses2d_batch, poses3d_batch, noise, lr, isTraining=True)
                loss_g, loss_repro, loss_cam, loss_reg, loss_enc = repnet.Gstep(sess, poses2d_batch, poses3d_batch, noise, lr, isTraining=True)

            if i % 500 == 0:
                posesout = []
                for eval in range(poses2d_eval.shape[0]//batchsize):
                    noise_val = np.zeros([batchsize, args.latent_dim])  # sample random noise
                    posespred, campred = repnet.inference(sess, poses2d_eval[eval * batchsize: (eval + 1) * batchsize],
                                                          poses3d_eval[eval * batchsize: (eval + 1) * batchsize], noise_val, lr)
                    posesout.append(posespred)
                posesout = np.vstack(posesout)
                val = 0
                for p in range(posesout.shape[0]):
                    val = val + 1000 * err_3dpe(poses3d_eval[p:p+1, :], posesout[p:p+1, :])
                val = val/posesout.shape[0]

                logger.info('Error: {0:.3f}, Loss_d: {1:.3f}, Loss_gp: {2:.3f}, Loss_g: {3:.3f}, '
                            'Loss_repro: {4:.3f}, Loss_cam: {5:.3f}'
                            .format(val, loss_d, loss_gp, loss_g, loss_repro, loss_cam))
            # if i % 1000 == 0 and i > 0:
            #     repnet.saver.save(sess, os.path.join(models_dir, 'checkpoint'), global_step=repnet.global_step)
            if val < best_val:
                best_val = val
                repnet.saver.save(sess, os.path.join(models_dir, 'checkpoint'), global_step=repnet.global_step)
        if epoch % 1 == 0 and epoch > 0:
            lr = lr * args.lr_decay
