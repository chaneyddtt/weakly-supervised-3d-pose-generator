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
import ops

parse = argparse.ArgumentParser()
parse.add_argument("--batchsize", help= "the batch size used in training", default=128, type = int)
parse.add_argument("--epochs", help="number of epochs during training", default=50, type = int)
parse.add_argument("--latent_dim", help="dimension of latent space", default=1024, type = int)
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

parse.add_argument("--sampling", help="set to true if generate samples", default=True, type=bool)
parse.add_argument("--checkpoint", help="which model to load", default=0, type=int)
# 931070 for gt data
# 971070 for shft
parse.add_argument("--num_samples", help="number of hypotheses", default=10, type=int)
parse.add_argument("--datatype", help="datatype used for training [GT, SHFT, GTMJ]", default='GT', type=str)
parse.add_argument("--load_path", help="specify the path to load model", default='./models', type=str)
args = parse.parse_args()


actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
pose3d_dim = 16 * 3
pose2d_dim = 16 * 2
cam_dim = 6
lr = args.lr
model_name = '{}_regweight{}_encweight{}_2D{}'.format(args.architecture, args.reg_weight, args.enc_weight, args.datatype)
log_dir = 'logs_eval'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.config.fileConfig('./logging.conf')
logger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/log.txt".format(log_dir))
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
mmd_posenet = Pose_mmdgan_enc(posenet, camnet, disc, encoder, args.latent_dim, args.batchsize, log_dir, args.epochs, pose2d_dim, pose3d_dim,
                 args.kernel, args.repro_weight, args.cam_weight, args.gp_weight, args.reg_weight, args.dot_weight, args.enc_weight)
mmd_posenet.build_model()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    batchsize = args.batchsize
    load_dir = os.path.join(args.load_path, model_name)
    ckpt = tf.train.get_checkpoint_state(load_dir, latest_filename="checkpoint")
    if args.checkpoint > 0:
        ckpt_name = os.path.join(os.path.join(load_dir, "checkpoint-{}".format(args.checkpoint)))
    else:
        ckpt_name = ckpt.model_checkpoint_path
    mmd_posenet.saver.restore(sess, ckpt_name)
    print('Loading model {}'.format(os.path.basename(ckpt_name)))

    path = 'new_data/test/2d{}_3dTEM'.format(args.datatype)
    path_cam = 'new_data/test/2d{}_3dCAM'.format(args.datatype)
    logger.info('{0:>15} {1:>30} {2:>30}'.format('Action', 'Protocol1', 'Protocol2'))
    val_best_all = []
    valcam_best_all = []

    val_zc_all = []
    valcam_zc_all = []

    for action in actions:
        data_2d_3d_test = sio.loadmat('{}/{}_2d{}_3d_test.mat'.format(path, action, args.datatype))
        data_cam = sio.loadmat('{}/{}_2d{}_3d_test.mat'.format(path_cam, action, args.datatype))
        poses2d_eval = data_2d_3d_test['poses_2d'][::64, :]
        poses3d_eval = data_2d_3d_test['poses_3d'][::64, :] / 1000
        poses_3d_cam = data_cam['poses_3d'][::64, :] / 1000

        poses_zc = []
        posescam_zc = []
        # generate results under zero code setting
        for eval in range(poses2d_eval.shape[0] // batchsize):
            noise_zc = np.zeros([batchsize, args.latent_dim])
            poses, cam = mmd_posenet.inference(sess, poses2d_eval[eval * batchsize: (eval + 1) * batchsize],
                                                  poses3d_eval[eval * batchsize: (eval + 1) * batchsize], noise_zc,
                                                  lr)
            poses_reshape = np.reshape(poses, [poses.shape[0], 3, 16])
            k = np.reshape(cam, [cam.shape[0], 2, 3])
            R = ops.compute_R(k)  # recover rotation matrix from camera matrix
            poses_cam = np.matmul(R, poses_reshape)  # transfer pose from the template frame to the camera frame
            poses_cam_reshape = np.reshape(poses_cam, [poses_cam.shape[0], -1])
            posescam_zc.append(poses_cam_reshape)
            poses_zc.append(poses)
        poses_zc = np.vstack(poses_zc)
        posescam_zc = np.vstack(posescam_zc)

        # compute the error under zero code setting
        val_zc = 0.0
        valcam_zc = 0.0
        for p in range(poses_zc.shape[0]):
            err_zc = 1000 * err_3dpe(poses3d_eval[p:p + 1, :], poses_zc[p:p + 1, :], True)
            errcam_zc = 1000 * err_3dpe(poses_3d_cam[p:p + 1, :], 1.1 * posescam_zc[p:p + 1, :], False)
            # scale the output according to the ratio between poses in camera frame and poses in template frame in the training set
            val_zc = val_zc + err_zc
            valcam_zc = valcam_zc + errcam_zc
            val_zc_all.append(err_zc)
            valcam_zc_all.append(errcam_zc)

        val_zc = val_zc / poses_zc.shape[0]
        valcam_zc = valcam_zc/posescam_zc.shape[0]

        # generate results for multiple hypotheses
        poses_samples_all = []
        posescam_samples_all = []
        R_all = []
        poses_repro_all = []
        for eval in range(poses2d_eval.shape[0] // batchsize):
            poses_samples_batch = []
            posescam_samples_batch = []
            poses_repro_batch = []
            for i in range(args.num_samples):
                z_test = np.random.normal(0, 1, (batchsize, args.latent_dim))
                posespred, campred = mmd_posenet.inference(sess, poses2d_eval[eval * batchsize: (eval + 1) * batchsize],
                                                      poses3d_eval[eval * batchsize: (eval + 1) * batchsize], z_test,
                                                      lr)

                posespred_reshape = np.reshape(posespred, [posespred.shape[0], 3, 16])
                poses_samples_batch.append(posespred)

                k = np.reshape(campred, [campred.shape[0], 2, 3])
                R = ops.compute_R(k)
                posespred_cam = np.matmul(R, posespred_reshape)
                posespred_cam_reshape = np.reshape(posespred_cam, [posespred_cam.shape[0], -1])
                posescam_samples_batch.append(posespred_cam_reshape)

                poses_repro = np.reshape(np.matmul(k, posespred_reshape), [posespred.shape[0], -1])
                poses_repro_batch.append(poses_repro)

            poses_samples_batch = np.stack(poses_samples_batch, axis=1)
            poses_samples_all.append(poses_samples_batch)

            posescam_samples_batch = np.stack(posescam_samples_batch,axis=1)
            posescam_samples_all.append(posescam_samples_batch)

            poses_repro_batch = np.stack(poses_repro_batch, axis=1)
            poses_repro_all.append(poses_repro_batch)

            R_all.append(R)

        poses_samples_all = np.concatenate(poses_samples_all, axis=0)
        posescam_samples_all = np.concatenate(posescam_samples_all, axis=0)
        poses_repro_all = np.concatenate(poses_repro_all, axis=0)
        R_all = np.concatenate(R_all, axis=0)

        # compute error for bh setting
        err = np.zeros([poses_samples_all.shape[0], poses_samples_all.shape[1]])
        err_cam = np.zeros([poses_samples_all.shape[0], poses_samples_all.shape[1]])
        for p in range(err.shape[0]):
            for s in range(args.num_samples):
                err[p, s] = 1000 * err_3dpe(poses3d_eval[p:p + 1, :], poses_samples_all[p:p + 1, s, :], True)
                err_cam[p, s] = 1000 * err_3dpe(poses_3d_cam[p:p + 1, :],  1.1 * posescam_samples_all[p:p + 1, s, :],
                                                False)  # scale the output according to the ratio between poses in camera
                                                        # frame and poses in template frame in the training set
        val_best = np.mean(np.min(err, axis=1))
        valcam_best = np.mean(np.min(err_cam, axis=1))
        val_best_all.append(np.min(err, axis=1))
        valcam_best_all.append(np.min(err_cam, axis=1))
        logger.info('{0:<15} {1:>15.2f} {2:>15.2f} {3:>15.2f} {4:>15.2f}'.format(action, valcam_zc, valcam_best, val_zc, val_best ))

    valcam_zc_all = np.array(valcam_zc_all)
    val_zc_all = np.array(val_zc_all)
    valcam_best_all = np.concatenate(valcam_best_all)
    val_best_all = np.concatenate(val_best_all)
    logger.info('{0:<15} {1:>15.2f} {2:>15.2f} {3:>15.2f} {4:>15.2f}'.format('Average', np.mean(valcam_zc_all), np.mean(valcam_best_all),
                                                                         np.mean(val_zc_all), np.mean(val_best_all)))

    # the result for each column represents: protocol 1 (zc, bh), protocol 2 (zc, bh)


