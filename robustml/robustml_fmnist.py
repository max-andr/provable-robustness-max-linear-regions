import sys
sys.path.append('..')
import argparse
import scipy.io
import tensorflow as tf
import robustml
import models


def load_model(sess, model, model_path):
    param_file = scipy.io.loadmat(model_path)
    weight_names = ['weights_conv1', 'weights_conv2', 'weights_fc1', 'weights_fc2']
    bias_names = ['biases_conv1', 'biases_conv2', 'biases_fc1', 'biases_fc2']

    for var_tf, var_name_mat in zip(model.W, weight_names):
        var_tf.load(param_file[var_name_mat], sess)
    for var_tf, var_name_mat in zip(model.b, bias_names):
        bias_val = param_file[var_name_mat]
        bias_val = bias_val.flatten()
        var_tf.load(bias_val, sess)


class Model(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        height, width, n_col = 28, 28, 1
        self._input = tf.placeholder(tf.float32, (height, width, n_col))  # assuming inputs in [0, 1]
        input_expanded = tf.expand_dims(self._input, axis=0)
        hps = argparse.Namespace(height=height, width=width, n_col=n_col)  # needed for models.LeNetSmall
        self._model = models.LeNetSmall(False, hps)
        self._logits = self._model.net(input_expanded)[-1]

        # load the model
        model_path = "models/mmr+at/2019-02-16 12:08:43 dataset=fmnist nn_type=cnn_lenet_small p_norm=inf lmbd=2.0 gamma_rb=0.15 gamma_db=0.15 stage1hpl=10 ae_frac=0.5 epoch=100.mat"
        load_model(sess, self._model, model_path)

        self._dataset = robustml.dataset.FMNIST()
        self._threat_model = robustml.threat_model.Linf(epsilon=0.1)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        logits_val = self._sess.run(self._logits, feed_dict={self._input: x})[0]
        pred_label = logits_val.argmax()  # label as a number
        return pred_label

