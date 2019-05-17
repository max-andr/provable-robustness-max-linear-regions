import argparse
import time
from datetime import datetime

import numpy as np
import os
import scipy.io
import tensorflow as tf
import torch
import torch.utils.data as td
from cleverhans.model import CallableModelWrapper

import attacks as ae
import data
import kolter_wong.eval as eval
import kolter_wong.models
import models
import utils
from mip import mip_verify

nn_dict = {'fc': models.MLP,
           'cnn_lenet_small': models.LeNetSmall,
           }
eps_dict = {2: {'mnist': 0.3,
                'fmnist': 0.3,
                'gts': 0.2,
                'cifar10': 0.1},
            np.inf: {'mnist': 0.1,
                     'fmnist': 0.1,
                     'gts': 4 / 255,
                     'cifar10': 2 / 255},
            }


def forward_pass(x):
    """ Only for compatibility with Cleverhans. """
    logits = model.net(x)[-1]
    return logits


def eval_in_batches(x_in, y_in, sess, tensors, batch_iter):
    """Get all predictions for a dataset by running it in small batches.
       Note, we assume that this is not for training, thus is_train=False
    """
    vals_total = [0] * len(tensors)
    n_batches = 0
    for batch_x, batch_y in batch_iter:
        vals = sess.run(tensors, feed_dict={x_in: batch_x, y_in: batch_y,
                                            is_train: False})
        for i in range(len(vals)):
            vals_total[i] += vals[i]
        n_batches += 1
    return [val_total / n_batches for val_total in vals_total]


def export_weights(sess, model, mat_path):
    weights_tf = model.W + model.b
    if hps.nn_type == 'fc1':
        W1_, W2_, b1_, b2_ = sess.run(weights_tf)
        scipy.io.savemat(mat_path, mdict={'U': W1_, 'W': W2_, 'bU': b1_, 'bW': b2_})
    else:
        var_val_dict = dict([(var.name.split(':')[0], val) for var, val in zip(weights_tf, sess.run(weights_tf))])
        scipy.io.savemat(mat_path, mdict=var_val_dict)


def load_model(sess, hps):
    """
    Note: in order to correctly load a model, one has to perform some reshapes or preprocessing steps.
    """
    param_file = scipy.io.loadmat(hps.model_path)
    if hps.nn_type == 'fc1':
        weight_names = ['U', 'W']
        bias_names = ['bU', 'bW']
    elif hps.nn_type == 'cnn_lenet_small':
        weight_names = ['weights_conv1', 'weights_conv2', 'weights_fc1', 'weights_fc2']
        bias_names = ['biases_conv1', 'biases_conv2', 'biases_fc1', 'biases_fc2']
    else:
        raise ValueError('wrong nn_type')

    for var_tf, var_name_mat in zip(model.W, weight_names):
        var_tf.load(param_file[var_name_mat], sess)
    for var_tf, var_name_mat in zip(model.b, bias_names):
        bias_val = param_file[var_name_mat]
        if hps.nn_type == 'cnn_lenet_small':
            bias_val = bias_val.flatten()
        var_tf.load(bias_val, sess)


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU indices.')
parser.add_argument('--exp_name', type=str, default='test',
                    help='Name of the experiment, which is used to save the results/metrics/model in a certain folder.')
parser.add_argument('--export_folder', type=str, default='eval_test', help='Where to export the evaluation results.')
parser.add_argument('--dataset', type=str, default='mnist', help='mnist, cifar10, fmnist, gts')
parser.add_argument('--nn_type', type=str, default='fc1', help='NN type: fc1, cnn_lenet_small')
parser.add_argument('--pgd_n_iter', type=int, default=40, help='e.g. 40, 100, etc')
parser.add_argument('--n_test_eval', type=int, default=10000, help='On how many examples to perform the evaluation.')
parser.add_argument('--model_path', type=str, default='', help='path to the model .mat file')
parser.add_argument('--p', type=str, default='inf', help='P-norm: 2 or inf')


hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10

hps.seed = 1
norm_str_mip = hps.p
hps.p = {'1': 1, '2': 2, 'inf': np.inf}[hps.p]
eps = eps_dict[hps.p][hps.dataset]

hps.gpu_memory = 0.1 if hps.dataset in ['mnist', 'fmnist'] else 0.15

cur_timestamp = str(datetime.now())[:-7]  # to get rid of milliseconds

log = utils.Logger()

eval_name = 'eval_{}_l{}'.format(hps.exp_name, norm_str_mip)  # e.g. eval_test_linf
hps.export_folder = 'models_eval/{}/'.format(eval_name)
utils.create_folders([hps.export_folder])

_, x_test, _, y_test = data.get_dataset(hps.dataset)
n_test_ex, hps.height, hps.width, hps.n_col = x_test.shape
hps.n_in, hps.n_out = hps.height * hps.width * hps.n_col, y_test.shape[1]

model_type = hps.nn_type if 'cnn' in hps.nn_type else 'fc'  # used to select the correct model from models.py
if hps.nn_type == 'fc1':
    hps.n_hs = [1024]
else:
    hps.n_hs = []

graph = tf.Graph()
with graph.as_default(), tf.device('/gpu:0'):
    x_in = tf.placeholder(tf.float32, [None, hps.height, hps.width, hps.n_col])
    y_in = tf.placeholder(tf.float32, [None, hps.n_out])
    is_train = tf.placeholder(tf.bool, name='is_training')
    lr_tf = tf.placeholder(tf.float32, shape=[])
    n_rb_tf = tf.placeholder(tf.int32, shape=[])
    n_db_tf = tf.placeholder(tf.int32, shape=[])

    x_tf = tf.identity(x_in)

    model = nn_dict[model_type](is_train, hps)
    cleverhans_model = CallableModelWrapper(forward_pass, 'logits')

    # Separate forward pass graph for Cleverhans wrapper (for C&W and PGD attacks) placed on the last GPU
    logits = forward_pass(x_tf)

    # Error rate
    incorrect_prediction = tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_in, 1))
    err_rate = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))

    # GPU settings
    gpu_options = tf.GPUOptions(visible_device_list=str(hps.gpus)[1:-1], per_process_gpu_memory_fraction=hps.gpu_memory)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

# ---------  Pytorch part for the Kolter-Wong model ----------
device = torch.device('cuda:' + str(hps.gpus[-1]))
torch.cuda.set_device(hps.gpus[-1])

model_torch = kolter_wong.models.select_model(hps.nn_type, hps.n_in, hps.n_out).to(device)
for var in model_torch.parameters():
    var.requires_grad = False
# ----------      end      ----------


with tf.Session(graph=graph, config=config) as sess:
    with graph.as_default(), tf.device('/gpu:0'):
        pgd_ae_tensor = ae.pgd_attack(x_tf, y_in, cleverhans_model, hps.p, eps, hps.pgd_n_iter)
    load_model(sess, hps)  # load the weights from hps.model_path to the current TF model

    time_start = time.time()
    # First on the full test set
    clean_inputs_all, clean_labels_all = x_test, y_test

    logits_val = sess.run(logits, feed_dict={x_in: clean_inputs_all, is_train: False})
    correctly_classified = np.argmax(clean_labels_all, axis=1) == np.argmax(logits_val, axis=1)
    test_error_tf = 1 - correctly_classified.mean()
    log.add('test err {:.2%}'.format(test_error_tf))

    # Evaluate the rest only on a subset
    clean_inputs_all, clean_labels_all = x_test[:hps.n_test_eval], y_test[:hps.n_test_eval]
    logits_val = sess.run(logits, feed_dict={x_tf: clean_inputs_all, is_train: False})
    correctly_classified = np.argmax(clean_labels_all, axis=1) == np.argmax(logits_val, axis=1)

    # ------------- Eval PGD -----------------------
    is_ae_misclassified = ae.eval_pgd_attack(sess, pgd_ae_tensor, x_tf, y_in, logits, is_train,
                                             clean_inputs_all, clean_labels_all)
    err_rate_pgd = np.mean(is_ae_misclassified)
    # -----------------------------------------------

    # --------------- Eval KW  -----------------------
    batch_size_kw = 10  # make sure that memory consumption is not too large
    n_batches_kw = n_test_ex // batch_size_kw

    X_te = torch.from_numpy(clean_inputs_all.transpose([0, 3, 1, 2])).float()  # to HCHW
    y_te = torch.from_numpy(clean_labels_all.argmax(1)).long()
    test_data = td.TensorDataset(X_te, y_te)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_kw)

    kolter_wong.models.restore_model(sess, model_torch, model, hps.nn_type, device)

    time_begin_lb = time.time()

    kw_dist_lb, kw_pred_correct = eval.eval_lb_db(hps.p, model_torch, test_loader, n_batches_kw, device)

    test_error_pt = 1 - kw_pred_correct.mean()
    avg_lb_lp_correct, avg_lb_lp_incorrect = kw_dist_lb[kw_pred_correct].mean(), kw_dist_lb[~kw_pred_correct].mean()
    # if lb > eps AND correctly classified  =>  we can guarantee the same class
    kw_dist_lb = kw_dist_lb * kw_pred_correct  # i.e. zero out the LBs with incorrect predictions
    kw_error_ub = 1 - (kw_dist_lb > eps).mean()

    lower_bounds_time = time.time() - time_begin_lb
    avg_lower_bounds = kw_dist_lb.mean()

    log.add('KW done in {:.2f} sec'.format(lower_bounds_time))

    model_name = hps.model_path.split('/')[-1].replace('.mat', '__exported.mat')  # take only the model name without its full path
    mat_path = hps.export_folder + model_name
    export_weights(sess, model, mat_path)  # export the model to the same folder as MIP results will be

if hps.p == np.inf:  # eval MIP
    mip_verify.verify(mat_path, hps.dataset, hps.nn_type, norm_str_mip, eps, hps.n_test_eval,
                      120, 'lp', time_limit_bounds=5)

    mip_error_clean, mip_error_lb, mip_error_ub, mip_frac_timeout, mip_total_time, mip_n_points_verified = \
        mip_verify.parse_summary(mat_path, hps.dataset, eps)
    mip_is_non_robust, mip_is_provably_robust = mip_verify.parse_bounds_pointwise(mat_path, hps.dataset, eps)

    # Combine pgd, kw and mip
    combined_non_robust = is_ae_misclassified + mip_is_non_robust
    combined_provably_robust = (kw_dist_lb > eps) + mip_is_provably_robust
    combined_err_lb = combined_non_robust.mean()
    combined_err_ub = 1 - combined_provably_robust.mean()

    log.add('Test err {:.2%}, error mip clean: {:.2%}, pgd lb: {:.2%}, mip lb: {:.2%}, pgd+mip lb: {:.2%}, '
            'kw+mip ub: {:.2%}, mip ub: {:.2%}, kw ub: {:.2%}, frac timeout {:.2%} ({} pts in {:.2f} min)'.
            format(test_error_tf, mip_error_clean, err_rate_pgd, mip_error_lb, combined_err_lb, combined_err_ub,
                   mip_error_ub, kw_error_ub, mip_frac_timeout, mip_n_points_verified, mip_total_time / 60))
    dict_export = {
        'mip_error_clean': mip_error_clean,
        'mip_is_non_robust': mip_is_non_robust, 'mip_is_provably_robust': mip_is_provably_robust,
        'mip_error_lb': mip_error_lb, 'mip_error_ub': mip_error_ub,
        'mip_total_time': mip_total_time,

        'combined_non_robust': combined_non_robust, 'combined_provably_robust': combined_provably_robust,
        'combined_err_lb': combined_err_lb, 'combined_err_ub': combined_err_ub,
    }
else:
    log.add(
        'test_error_tf {:.2%}  test_err_pt {:.2%}  error lb pgd: {:.2%}  error ub kw: {:.2%} (eps={:.3f}, {:.2f} sec elapsed)'.format(
            test_error_tf, test_error_pt, err_rate_pgd, kw_error_ub, eps, time.time() - time_start))
    dict_export = {}

dict_export.update({'dataset': hps.dataset, 'nn_type': hps.nn_type,
                    'correctly_classified_clean': correctly_classified,
                    'is_ae_misclassified_pgd': is_ae_misclassified,

                    'kw_dist_lb': kw_dist_lb, 'kw_dist_lb_avg': avg_lower_bounds,
                    'lower_bounds_time': lower_bounds_time,

                    'test_error_tf': test_error_tf, 'test_error_pt': test_error_pt
                    })

export_file_results = '{}/{}__{}'.format(hps.export_folder, hps.exp_name, model_name)

scipy.io.savemat(export_file_results, mdict=dict_export)

log.add('Eval done in {:.2f} min ({})\n'.format((time.time() - time_start) / 60, hps.model_path))
