"""
Unit test for 1 epoch on MNIST for a given architecture assumming random seed=1.
"""
import tensorflow as tf
import subprocess
import argparse
import os


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--nn_type', type=str, default='fc1', help='fc1, fc10, fc10_large')
parser.add_argument('--reg', type=str, default='full', help='full, cheap')
parser.add_argument('--dataset', type=str, default='mnist', help='only mnist is supported now')
parser.add_argument('--gpus', type=str, default='0', help='0, 1, ..., 7')
hps = parser.parse_args()
exp_name = 'unit_test'

print('Unit test is started')
subprocess.call("python train.py --exp_name={} --gpus={} --lmbd=1.5 --gamma_rb=1.0 --gamma_db=0.5 "
                "--ae_frac=0.0 --n_epochs=0 --dataset={} --nn_type={} --reg={} --n_cw=0 --no_mat_eval".
                format(exp_name, hps.gpus, hps.dataset, hps.nn_type, hps.reg), shell=True)

# Read test error, test reg, train error, train reg
unit_test_dir = 'exps/{}/tb/{}'.format(exp_name, hps.dataset)
run_name = sorted(os.listdir(unit_test_dir))[-1]  # select the latest run

max_epoch = 0
cur_vals = []
for mode in ['test', 'train']:
    events_fname = os.listdir(unit_test_dir + '/' + run_name + '/' + mode)[0]
    for e in tf.train.summary_iterator(unit_test_dir + '/' + run_name + '/' + mode + '/' + events_fname):
        for v in e.summary.value:
            if v.tag in ['main/loss_plain', 'main/reg']:
                cur_vals.append(v.simple_value)

# Generate test report
metrics = ['test_err', 'test_reg', 'train_err', 'train_reg']
true_vals_dict = {'mnist__fc1': [2.373158, 2.88881, 2.370359, 2.88442],  # for 10+10 hpl regularizer at 0th epoch
                  'mnist__fc10': [2.297094, 2.82142, 2.297866, 2.82542]  # for 10+10 hpl regularizer at 0th epoch
                  }
true_vals = true_vals_dict[hps.dataset + '__' + hps.nn_type]
print('metric: true_val cur_val diff')
for true_val, cur_val, metric in zip(true_vals, cur_vals, metrics):
    print('{}: {:.5f} {:.5f} {:.5f}'.format(metric, true_val, cur_val, true_val - cur_val))
print('Unit test is done')
