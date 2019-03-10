import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from cleverhans.model import CallableModelWrapper

import attacks as ae
import data
import models
import regularizers
import utils

nn_dict = {'fc': models.MLP,
           'cnn_lenet_small': models.LeNetSmall
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


def adv_train(x, y):
    n_adv = int(hps.ae_frac * hps.batch_size)
    if n_adv == 0:
        return x
    else:
        adv_inputs = ae.pgd_attack(x[:n_adv], y[:n_adv], cleverhans_model, hps.p,
                                   eps_dict[hps.p][hps.dataset], hps.pgd_n_iter)
        return tf.concat([adv_inputs, x[n_adv:]], axis=0)


def forward_pass_cleverhans(x):
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
        vals = sess.run(tensors, feed_dict={x_in: batch_x, y_in: batch_y, lr_tf: hps.lr,
                                            n_rb_tf: n_rb, n_db_tf: n_db, frac_reg_tf: frac_reg,
                                            is_train: False})
        for i in range(len(vals)):
            vals_total[i] += vals[i]
        n_batches += 1
    return [val_total / n_batches for val_total in vals_total]


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU indices. Multi-gpu training is supported.')
parser.add_argument('--gpu_memory', type=float, default=0.0,
                    help='GPU memory fraction to use')
parser.add_argument('--exp_name', type=str, default='test',
                    help='Name of the experiment, which is used to save the results/metrics/model in a certain folder.')
parser.add_argument('--dataset', type=str, default='mnist', help='mnist, cifar10, fmnist, gts, svhn')
parser.add_argument('--nn_type', type=str, default='fc1', help='NN type: fc1, fc10, cnn_lenet_small')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs.')
parser.add_argument('--reg', type=str, default='full', help='full, cheap')
parser.add_argument('--lmbd', type=float, default=1.0, help='Lambda')
parser.add_argument('--gamma_rb', type=float, default=1.0, help='Gamma for region boundaries')
parser.add_argument('--gamma_db', type=float, default=1.0, help='Gamma decision boundaries')
parser.add_argument('--ae_frac', type=float, default=0.0, help='Fraction of AE in a batch [0..1].')
parser.add_argument('--restore', action='store_true',
                    help='Restore the model by exp_name+dataset and combination of lambda and gammas')
parser.add_argument('--data_augm', action='store_true',
                    help='Data augmentation: rotation, mirroring (not for mnist and gtrsrb), gauss noise.')
parser.add_argument('--p', type=str, default='2', help='P-norm: 2 or inf (note: as strings)')

hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10
hps.batch_size = 128
hps.lr = 0.001
# mnist and fmnist with L2-MMR require a smaller learning rate
if hps.p == '2' and hps.dataset in ['mnist', 'fmnist'] and hps.lmbd > 0.0:
    if hps.ae_frac == 0.0:
        hps.lr = 0.0001
    else:  # if AT
        hps.lr = 0.00005
hps_str = utils.create_hps_str(hps)
hps.seed = 1
hps.p = {'1': 1, '2': 2, 'inf': np.inf}[hps.p]
hps.q = {1: np.inf, 2: 2, np.inf: 1}[hps.p]  # q norm is used in the denominator of MMR
if hps.p == np.inf and hps.gamma_db >= 1.0:  # supports 2 ways of specifying gammas: in [0..1] and in [1..255]
    hps.gamma_rb, hps.gamma_db = hps.gamma_rb / 255, hps.gamma_db / 255  # to make it easier to specify gammas via cmd

hps.pgd_n_iter = 40 if hps.dataset in ['mnist', 'fmnist'] else 7  # as in Madry et al

if hps.gpu_memory == 0.0:  # if it wasn't set
    if 'cnn' in hps.nn_type:
        hps.gpu_memory = 0.85
    elif hps.dataset in ['mnist', 'fmnist'] or hps.lmbd == 0.0:
        hps.gpu_memory = 0.15
    else:
        hps.gpu_memory = 0.25

cur_timestamp = str(datetime.now())[:-7]  # to get rid of milliseconds

log = utils.Logger()
log.add('The script started on GPUs {} with hyperparameters: {} at {}'.format(hps.gpus, hps_str, cur_timestamp))

x_train, x_test, y_train, y_test = data.get_dataset(hps.dataset)  # e.g. x_train of mnist has (60000, 28, 28, 1) shape
n_test_ex = x_test.shape[0]
n_train_ex, hps.height, hps.width, hps.n_col = x_train.shape
hps.n_in, hps.n_out = hps.height * hps.width * hps.n_col, y_train.shape[1]
if hps.data_augm:
    n_pad = 2  # n_pad pixels from very side (increases the height and width by 2*n_pad)
    hps.height_pad, hps.width_pad = hps.height + 2 * n_pad, hps.width + 2 * n_pad
    x_train, x_test = data.zero_pad(x_train, n_pad), data.zero_pad(x_test, n_pad)
    hps.random_crop, hps.fl_mirroring, hps.gauss_noise_flag = True, True, False
    hps.fl_rotations, hps.max_rotate_angle = False, np.pi / 20

model_type = hps.nn_type if 'cnn' in hps.nn_type else 'fc'  # is used to select the correct model from models.py

hps.n_hs = utils.get_hidden_units(hps.nn_type)
n_batches_train = n_train_ex // hps.batch_size  # 'all' is 600 with bs=100 or 750 with bs=80
n_batches_test = n_test_ex // hps.batch_size
n_eval_pgd = 200
n_eval_pgd_final = 10000

eps_pgd = eps_dict[hps.p][hps.dataset]

graph = tf.Graph()
with graph.as_default(), tf.device('/gpu:0'):
    if hps.data_augm:
        x_in = tf.placeholder(tf.float32, [None, hps.height_pad, hps.width_pad, hps.n_col])
    else:
        x_in = tf.placeholder(tf.float32, [None, hps.height, hps.width, hps.n_col])
    y_in = tf.placeholder(tf.float32, [None, hps.n_out])
    is_train = tf.placeholder(tf.bool, name='is_training')
    lr_tf = tf.placeholder(tf.float32, shape=[])
    frac_reg_tf = tf.placeholder(tf.float32, shape=[])  # from 0 to 1 - how strong is the regularizer
    n_rb_tf = tf.placeholder(tf.int32, shape=[])
    n_db_tf = tf.placeholder(tf.int32, shape=[])

    if hps.data_augm:  # Data augmentation is implemented inside the TF comp. graph
        x_tf = tf.cond(is_train, lambda: data.augment_train(x_in, hps),
                       lambda: data.augment_test(x_in, hps))
    else:
        x_tf = tf.identity(x_in)

    optimizer = tf.train.AdamOptimizer(lr_tf, beta1=0.9, beta2=0.999, epsilon=1e-08)

    tower_grads = []
    hps.batch_size_gpu = hps.batch_size // len(hps.gpus)
    losses_plain, losses_reg, regs_rb, regs_db, err_rates = [], [], [], [], []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(hps.gpus)):
            with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:

                id_from, id_to = i * hps.batch_size_gpu, i * hps.batch_size_gpu + hps.batch_size_gpu
                x, y = x_tf[id_from:id_to], y_in[id_from:id_to]
                log.add('Batch on GPU {}: from {} to {}, tensor {}'.format(i, id_from, id_to, x))

                model = nn_dict[model_type](is_train, hps)
                cleverhans_model = CallableModelWrapper(forward_pass_cleverhans, 'logits')

                # adv. training with PGD attack
                x = tf.cond(is_train, lambda: adv_train(x, y), lambda: x)

                y_list = model.net(x)

                ce_loss_per_example = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_list[-1])
                loss_plain_tower = tf.reduce_mean(ce_loss_per_example)
                if hps.lmbd > 0:  # for computational efficiency
                    # Regularizer components
                    if 'fc' in hps.nn_type:
                        rb_term, db_term = regularizers.mmr_fc(y_list, y, model.W, hps.n_in, hps.n_hs, hps.n_out,
                                                               n_rb_tf, n_db_tf, hps.gamma_rb, hps.gamma_db,
                                                               hps.batch_size_gpu, hps.q)
                    else:
                        rb_term, db_term = regularizers.mmr_cnn(y_list, x, y, model, n_rb_tf, n_db_tf, hps.gamma_rb,
                                                                hps.gamma_db, hps.batch_size_gpu, hps.q)
                    rb_reg_part = frac_reg_tf * hps.lmbd * rb_term / tf.cast(n_rb_tf, tf.float32)
                    db_reg_part = frac_reg_tf * hps.lmbd * db_term / tf.cast(n_db_tf, tf.float32)
                else:
                    rb_reg_part, db_reg_part = tf.constant(0.0), tf.constant(0.0)
                reg_rb_tower, reg_db_tower = tf.reduce_mean(rb_reg_part), tf.reduce_mean(db_reg_part)

                loss_reg_tower = loss_plain_tower + reg_rb_tower + reg_db_tower

                # Error rate
                incorrect_prediction = tf.not_equal(tf.argmax(y_list[-1], 1), tf.argmax(y, 1))
                err_rate_tower = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))

                # Calculate the gradients for the batch of data on this tower.
                grads_vars_in_tower = optimizer.compute_gradients(loss_reg_tower)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads_vars_in_tower)
                losses_plain.append(loss_plain_tower)
                losses_reg.append(loss_reg_tower)
                regs_rb.append(reg_rb_tower)
                regs_db.append(reg_db_tower)
                err_rates.append(err_rate_tower)

                # Reuse variables for this tower.
                tf.get_variable_scope().reuse_variables()

    loss_plain = tf.reduce_mean(tf.stack(losses_plain))
    loss_reg = tf.reduce_mean(tf.stack(losses_reg))
    reg_rb = tf.reduce_mean(tf.stack(regs_rb))
    reg_db = tf.reduce_mean(tf.stack(regs_db))
    err_rate = tf.reduce_mean(tf.stack(err_rates))

    grads_vars = utils.average_gradients(tower_grads)
    train_step_loss_reg = optimizer.apply_gradients(grads_vars, name='train_step')

    # Separate forward pass graph for Cleverhans wrapper (for PGD attack) placed on the last GPU
    logits_all_gpus = forward_pass_cleverhans(x_tf)

    # Model saver
    saver = tf.train.Saver()
    # GPU settings
    gpu_options = tf.GPUOptions(visible_device_list=str(hps.gpus)[1:-1], per_process_gpu_memory_fraction=hps.gpu_memory)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)


with tf.Session(graph=graph, config=config) as sess:
    with graph.as_default(), tf.device('/gpu:0'):
        pgd_ae_tensor = ae.pgd_attack(x_tf, y_in, cleverhans_model, hps.p, eps_dict[hps.p][hps.dataset], hps.pgd_n_iter)

    sess.run(tf.global_variables_initializer())  # run 'init' op
    epoch_start, epoch_end = 0, hps.n_epochs

    log.add('Session started with hyperparameters: {} \n'.format(hps_str))
    time_start = time.time()

    # for epoch in range(epoch_start, epoch_end):
    for epoch in range(1, epoch_end + 1):
        epoch_start_reduced_lr = 0.9
        lr_actual = hps.lr / 10 if epoch >= epoch_start_reduced_lr * hps.n_epochs else hps.lr

        frac_reg = min(epoch / 10.0, 1.0)  # from 0 to 1 linearly over the first 10 epochs

        frac_start, frac_end = 0.1, 0.02  # decrease the number of linear region hyperplanes from 10% to 2%
        n_db = hps.n_out  # the number of decision boundary hyperplanes is always the same (the number of classes)
        n_total_hidden_units = utils.get_n_total_hidden_units(hps.nn_type, hps.n_hs, hps.height)
        n_rb_start, n_rb_end = int(frac_start * n_total_hidden_units), int(frac_end * n_total_hidden_units)
        n_rb = (n_rb_end - n_rb_start) / hps.n_epochs * epoch + n_rb_start

        tensors_to_eval = [err_rate, loss_plain, reg_rb, reg_db]
        err_rate_train, loss_plain_train, reg_rb_train, reg_db_train = 0, 0, 0, 0
        for batch_x, batch_y in data.get_batch_iterator(x_train, y_train, hps.batch_size, shuffle=True,
                                                        n_batches=n_batches_train):
            _, err_rate_train_val, loss_plain_train_val, reg_rb_val, reg_db_val = sess.run(
                [train_step_loss_reg] + tensors_to_eval, feed_dict={x_in: batch_x, y_in: batch_y, lr_tf: lr_actual,
                                                                    n_rb_tf: n_rb, n_db_tf: n_db,
                                                                    frac_reg_tf: frac_reg, is_train: True})
            err_rate_train += err_rate_train_val / n_batches_train
            loss_plain_train += loss_plain_train_val / n_batches_train
            reg_rb_train += reg_rb_val / n_batches_train
            reg_db_train += reg_db_val / n_batches_train

        log.add('Epoch {}: epoch training is done, {:.2f} sec elapsed'.format(epoch, time.time() - time_start))

        test_data_iter = data.get_batch_iterator(x_test, y_test, hps.batch_size, n_batches=n_batches_test)
        err_rate_test, loss_plain_test, reg_rb_test, reg_db_test = eval_in_batches(
            x_in, y_in, sess, tensors_to_eval, test_data_iter)
        reg_train, reg_test = reg_rb_train + reg_db_train, reg_rb_test + reg_db_test
        log.add('Epoch {}: train/test eval is done, {:.2f} sec elapsed'.format(epoch, time.time() - time_start))

        str_test = 'test_err {:.3%}  test_losspl {:.6f}  test_reg {:.5f}  '.format(
            err_rate_test, loss_plain_test, reg_test)
        str_train = 'train_err {:.3%}  train_losspl {:.6f}  train_reg {:.5f}  '.format(
            err_rate_train, loss_plain_train, reg_train)
        log.add('Epoch {:d}  '.format(epoch) + str_test + str_train)

        # PGD evaluation
        if epoch == epoch_end:  # last epoch - full test set evaluation
            clean_inputs_all, clean_labels_all = x_test[:n_eval_pgd_final], y_test[:n_eval_pgd_final]
        else:
            clean_inputs_all, clean_labels_all = x_test[:n_eval_pgd], y_test[:n_eval_pgd]

        is_ae_misclassified = ae.eval_pgd_attack(sess, pgd_ae_tensor, x_tf, y_in, logits_all_gpus, is_train,
                                                 clean_inputs_all, clean_labels_all)
        adv_error_pgd = np.mean(is_ae_misclassified)
        log.add('Epoch {}:  PGD eps={:.3f} err_rate={:.2%}'.format(epoch, eps_pgd, adv_error_pgd))

        save_model_every_n_epochs = 10
        if (epoch % save_model_every_n_epochs == 0 and epoch != epoch_start) or epoch == epoch_end:
            mat_path, bounds_path = utils.save_results(sess, saver, model.W + model.b, cur_timestamp, hps_str, hps,
                                                       log, epoch)

log.add('Worker done in {:.2f} min ({})\n\n'.format((time.time() - time_start) / 60, hps_str))

