import numpy
import numpy as np
import scipy.io
import tensorflow as tf


def get_batch_iterator(X, Y, batch_size, shuffle=False, n_batches='all'):
    if n_batches == 'all':
        n_batches = len(X) // batch_size
    if shuffle:
        x_idx = np.random.permutation(len(X))[:n_batches * batch_size]
    else:
        x_idx = np.arange(len(X))[:n_batches * batch_size]
    for batch_idx in x_idx.reshape([len(x_idx) // batch_size, batch_size]):
        batch_x, batch_y = X[batch_idx], Y[batch_idx]
        yield batch_x, batch_y


def get_dataset(dataset):
    # Originally, all pixels values are uint8 values in [0, 255]
    train = scipy.io.loadmat('datasets/{}/{}_int_train.mat'.format(dataset, dataset))
    test = scipy.io.loadmat('datasets/{}/{}_int_test.mat'.format(dataset, dataset))
    x_train, y_train, x_test, y_test = train['images'], train['labels'], test['images'], test['labels']

    y_train, y_test = dense_to_one_hot(y_train.flatten()), dense_to_one_hot(y_test.flatten())
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if dataset == 'gts':  # we drop the last partial batch
        x_test, y_test = x_test[:12600], y_test[:12600]

    return x_train, x_test, y_train, y_test


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = len(np.unique(labels_dense))
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def zero_pad(x_img, n_pad):
    return np.pad(x_img, [(0, 0), (n_pad, n_pad), (n_pad, n_pad), (0, 0)], 'constant')


def augment_train(img_tensor, hps):
    def augment_each(img):
        if hps.random_crop:
            img = tf.random_crop(img, [hps.height, hps.width, hps.n_col])
        else:
            img = tf.image.central_crop(img, hps.height / hps.height_pad)
        if hps.dataset not in ['mnist', 'gts', 'svhn'] and hps.fl_mirroring:
            img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.minimum(tf.maximum(img, 0.0), 1.0)
        img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
        img = tf.minimum(tf.maximum(img, 0.0), 1.0)
        return img

    with tf.device('/cpu:0'):
        if hps.fl_rotations:
            rand_angles = tf.random_uniform([hps.batch_size], minval=-hps.max_rotate_angle, maxval=hps.max_rotate_angle)
            img_tensor = tf.contrib.image.rotate(img_tensor, rand_angles)
        img_tensor = tf.map_fn(augment_each, img_tensor)

        if hps.gauss_noise_flag:
            expected_noise_norm = 2.0 if hps.dataset in ['mnist', 'fmnist'] else 1.0
            gauss_noise = tf.random_normal(tf.shape(img_tensor), stddev=expected_noise_norm / hps.n_in)
            img_tensor += gauss_noise
            img_tensor = tf.minimum(tf.maximum(img_tensor, 0.0), 1.0)
        return img_tensor


def augment_test(img_tensor, hps):
    def prepare_each(img):
        img = tf.image.central_crop(img, hps.height / hps.height_pad)  # crop factor is 28/32 or 32/36
        return img

    with tf.device('/cpu:0'):
        img_tensor = tf.map_fn(prepare_each, img_tensor)
        return img_tensor

