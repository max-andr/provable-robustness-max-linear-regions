"""
Various helping functions
"""
import math
import pickle

import numpy as np
import os
import scipy.io
import tensorflow as tf

from mip import mip_verify


class Logger:
    def __init__(self):
        self.lst_this_run = []
        self.lst_whole_exp = []

    def add(self, string):
        self.lst_this_run.append(string)
        print(string)

    def clear(self):
        self.lst_this_run = []

    def to_file(self, folder, this_run_file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if this_run_file is not None:
            with open(folder + this_run_file, 'w') as f:
                f.write('\n'.join(self.lst_this_run))


def create_folders(folders):
    for folder in folders:
        current_folder = ''
        for component in folder.split('/')[:-1]:  # the last element of the list is ''
            current_folder += component + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)


def create_hps_str(hps):
    # We can't take all hps for file names, so we select the most important ones
    hyperparam_str = ("dataset={} nn_type={} p_norm={} lmbd={} gamma_rb={} gamma_db={} "
                      "ae_frac={}"). \
        format(hps.dataset, hps.nn_type, hps.p, hps.lmbd, hps.gamma_rb, hps.gamma_db,
               hps.ae_frac)
    return hyperparam_str


def save_results(sess, saver, weights, cur_timestamp, hps_str, hps, log, epoch):
    # Example of the folder structure: exps/cnn_linf/mat/mnist/
    base_path = '.'
    logs_path = '{}/exps/{}/{}/{}/'.format(base_path, hps.exp_name, 'logs', hps.dataset)
    models_path = '{}/exps/{}/{}/{}/'.format(base_path, hps.exp_name, 'models', hps.dataset)
    mat_path = '{}/exps/{}/{}/{}/'.format(base_path, hps.exp_name, 'mat', hps.dataset)
    bounds_path = '{}/exps/{}/{}/{}/'.format(base_path, hps.exp_name, 'bounds', hps.dataset)
    file_name = '{} {} epoch={}'.format(cur_timestamp, hps_str, epoch)

    create_folders([logs_path, models_path, mat_path, bounds_path])
    saver.save(sess, models_path + file_name)

    log.to_file(logs_path, file_name)

    # Export the weights
    if hps.nn_type == 'fc1':
        W1_, W2_, b1_, b2_ = sess.run(weights)
        scipy.io.savemat(mat_path + file_name, mdict={'U': W1_, 'W': W2_, 'bU': b1_, 'bW': b2_})
    else:
        vars = weights  # tf.trainable_variables()
        var_val_dict = dict([(var.name.split(':')[0], val) for var, val in zip(vars, sess.run(vars))])
        scipy.io.savemat(mat_path + file_name, mdict=var_val_dict)
    return mat_path + file_name + '.mat', bounds_path + file_name


def save_combined_bounds(epoch, eps_mip, mip, n_eval_mip):
    is_ae_misclassified, kw_dist_lb = mip[epoch]['is_ae_misclassified'], mip[epoch]['kw_dist_lb']
    mip_is_non_robust, mip_is_provably_robust = mip[epoch]['is_non_robust'], mip[epoch]['is_provably_robust']

    combined_non_robust = is_ae_misclassified[:n_eval_mip] + mip_is_non_robust
    combined_provably_robust = (kw_dist_lb[:n_eval_mip] >= eps_mip) + mip_is_provably_robust
    combined_err_lb = combined_non_robust.mean()
    combined_err_ub = 1 - combined_provably_robust.mean()

    print('Combined err lb {:.2%}, combined err ub {:.2%}'.format(combined_err_lb, combined_err_ub))
    scipy.io.savemat(mip[epoch]['bounds_path'],
                     mdict={'is_ae_misclassified': is_ae_misclassified, 'kw_dist_lb': kw_dist_lb,
                            'mip_is_non_robust': mip_is_non_robust, 'mip_is_provably_robust': mip_is_provably_robust})


def avg_tensor_list(tensor_list):
    tensors = tf.stack(axis=0, values=tensor_list)
    return tf.reduce_mean(tensors, axis=0)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_n_total_hidden_units(nn_type, n_hs, height_in):
    h1 = math.ceil(height_in / 2)
    h2 = math.ceil(h1 / 2)
    if 'fc' in nn_type:
        return np.sum(n_hs)
    elif nn_type == 'cnn_lenet_small':
        return 16 * h1 ** 2 + 32 * h2 ** 2 + 100  # MNIST/FMNIST: 4804 or CIFAR-10 / GTS: 6244
    # elif nn_type == 'cnn_large':
    else:
        raise ValueError('wrong arch for get_n_total_hidden_units()')


def get_hidden_units(nn_type):
    if nn_type == 'fc1':
        n_hs = [1024]
    elif nn_type == 'fc10':
        n_hs = [124, 104, 104, 104, 104, 104, 104, 104, 86, 86]
    elif 'cnn_lenet' in nn_type:
        n_hs = []  # not used at all
    else:
        n_hs = []
    n_hs = [int(v) for v in n_hs]
    return n_hs
