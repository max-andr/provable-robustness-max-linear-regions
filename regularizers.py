import numpy as np
import tensorflow as tf


def get_min_distances(dists, k):
    """
    dists: bs x #neurons - tensor of distances to hyperplanes
    k: int - how many hyperplanes to take
    """
    return -tf.nn.top_k(-dists, k=k).values


def zero_out_non_min_distances(dist, n_boundaries):
    n_units = dist.shape[1]
    dist = dist + 10**-5 * (np.random.rand(n_units) - 0.5)  # to break ties and don't count more than k zeros in CNNs
    min_dist_rb = get_min_distances(dist, n_boundaries)  # bs x n_boundaries
    th1 = tf.expand_dims(tf.reduce_max(min_dist_rb, axis=1), 1)  # bs x 1  -  take the maximum distance over min-k
    th1 = tf.tile(th1, [1, n_units])
    th = tf.cast(tf.less_equal(dist, th1), tf.float32)  # only entries that are <= max over min-k are 1, else 0
    return th


def calc_v_fc(V_prev, W):
    """
    V_prev: bs x d x n_prev - previous V matrix
    W: n_prev x n_next - last weight matrix
    relu: bs x n_prev - matrix of relu switches (0s and 1s) which are fixed for a particular input
    """
    n_in = V_prev.shape[1]
    n_prev, n_next = W.shape

    V = tf.reshape(V_prev, [-1, n_prev])  # bs*d x n_prev
    V = V @ W  # bs*d x n_next
    V = tf.reshape(V, [-1, n_in, n_next])  # bs x d x n_next
    return V


def calc_v_conv(V_prev, w, stride, padding):
    """
    V_prev: bs x d x h_prev x w_prev x c_prev  - previous V matrix
    w: h_filter x w_filter x c_prev x c_next - next conv filter
    relu: bs x h_prev x w_prev x c_prev - tensor of relu switches (0s and 1s) which are fixed for a particular input
    """
    d, h_prev, w_prev, c_prev = int(V_prev.shape[1]), int(V_prev.shape[2]), int(V_prev.shape[3]), int(V_prev.shape[4])
    V = tf.reshape(V_prev, [-1, h_prev, w_prev, c_prev])  # bs*d x h_prev x w_prev x c_prev
    V = tf.nn.conv2d(V, w, strides=[1, stride, stride, 1], padding=padding)  # bs*d x h_next x w_next x c_next
    V = tf.reshape(V, [-1, d, V.shape[1], V.shape[2], V.shape[3]])  # bs x d x h_next x w_next x c_next
    return V


def mmr_cnn(z_list, x, y_true, model, n_rb, n_db, gamma_rb, gamma_db, bs, q):
    """
    Batch-wise implementation of the Maximum Margin Regularizer for CNNs as a TensorFlow computational graph.
    Note that it is differentiable, and thus can be directly added to the main objective (e.g. the cross-entropy loss).

    z_list: list with all tensors that correspond to preactivation feature maps
            (in particular, z_list[-1] are logits; see models.LeNetSmall for details)
    x: input points (bs x image_height x image_width x image_n_channels)
    y_true: one-hot encoded ground truth labels (bs x n_classes)
    model: models.CNN object that contains a model with its weights, strides, padding, etc
    n_rb: number of closest region boundaries to take
    n_db: number of closest decision boundaries to take
    gamma_rb: gamma for region boundaries (approx. corresponds to the radius of the Lp-ball that we want to
              certify robustness in)
    gamma_db: gamma for decision boundaries (approx. corresponds to the radius of the Lp-ball that we want to
              be robust in)
    bs: batch size
    q: q-norm which is the dual norm to the p-norm that we aim to be robust at (e.g. if p=np.inf, q=1)
    """

    eps_num_stabil = 1e-5  # epsilon for numerical stability in the denominator of the distances

    # the padding and strides should be the same as in the forward pass conv
    strides, padding = model.strides, model.padding
    y_pred = z_list[-1]
    z_conv, z_fc, relus_conv, relus_fc, W_conv, W_fc = [], [], [], [], [], []
    for w, y in zip(model.W, z_list):  # Depending on the shape we form pre-activation values and their relu switches
        if len(y.shape) == 4:  # if conv layer
            z_conv.append(y)
            relu = tf.cast(tf.greater(y, 0), tf.float32)
            relus_conv.append(tf.expand_dims(relu, 1))
            W_conv.append(w)
        else:
            z_fc.append(y)
            relu = tf.cast(tf.greater(y, 0), tf.float32)
            relus_fc.append(tf.expand_dims(relu, 1))
            W_fc.append(w)

    h_in, w_in, c_in = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    n_in = h_in * w_in * c_in
    n_out = y_true.shape[1]

    # z[0]: bs x h_next x w_next x n_next,  W[0]: h_filter x w_filter x n_prev x n_next
    w_matrix = tf.reshape(W_conv[0], [-1, int(W_conv[0].shape[-1])])  # h_filter*w_filter*n_col x n_next
    denom = tf.norm(w_matrix, axis=0, ord=q, keep_dims=True)  # n_next
    dist_rb = tf.abs(z_conv[0]) / denom  # bs x h_next x w_next x n_next
    dist_rb = tf.reshape(dist_rb, [bs, int(z_conv[0].shape[1]*z_conv[0].shape[2]*z_conv[0].shape[3])])  # bs x h_next*w_next*n_next

    # We need to get the conv matrix. Instead of using loops to contruct such matrix, we can apply W[0] conv filter
    # to a reshaped identity matrix. Then we duplicate bs times the resulting tensor.
    identity_input_fm = tf.reshape(tf.eye(n_in, n_in), [1, n_in, h_in, w_in, c_in])
    V = calc_v_conv(identity_input_fm, W_conv[0], strides[0], padding)  # 1 x d x h_next x w_next x c_next
    V = tf.tile(V, [bs, 1, 1, 1, 1])  # bs x d x h_next x w_next x c_next
    V = V * relus_conv[0]
    for i in range(1, len(z_conv)):
        V = calc_v_conv(V, W_conv[i], strides[i], padding)  # bs x d x h_next x w_next x c_next
        V_stable = V + eps_num_stabil * tf.cast(tf.less(tf.abs(V), eps_num_stabil), tf.float32)  # note: +eps would also work
        new_dist_rb = tf.abs(z_conv[i]) / tf.norm(V_stable, axis=1, ord=q)  # bs x h_next x w_next x c_next
        new_dist_rb = tf.reshape(new_dist_rb, [bs, z_conv[i].shape[1]*z_conv[i].shape[2]*z_conv[i].shape[3]])  # bs x h_next*w_next*c_next
        dist_rb = tf.concat([dist_rb, new_dist_rb], 1)  # bs x sum(n_neurons[1:i])
        V = V * relus_conv[i]  # element-wise mult using broadcasting, result: bs x d x h_cur x w_cur x c_cur

    # Flattening after the last conv layer
    V = tf.reshape(V, [bs, n_in, V.shape[2] * V.shape[3] * V.shape[4]])  # bs x d x h_prev*w_prev*c_prev

    for i in range(len(z_fc) - 1):  # the last layer requires special handling
        V = calc_v_fc(V, W_fc[i])  # bs x d x n_hs[i]
        V_stable = V + eps_num_stabil * tf.cast(tf.less(tf.abs(V), eps_num_stabil), tf.float32)
        new_dist_rb = tf.abs(z_fc[i]) / tf.norm(V_stable, axis=1, ord=q)  # bs x n_hs[i]
        dist_rb = tf.concat([dist_rb, new_dist_rb], 1)  # bs x sum(n_hs[1:i])
        V = V * relus_fc[i]  # element-wise mult using broadcasting, result: bs x d x n_cur

    th = zero_out_non_min_distances(dist_rb, n_rb)
    rb_term = tf.reduce_sum(th * tf.maximum(0.0, 1.0 - dist_rb / gamma_rb), axis=1)

    # decision boundaries
    V = calc_v_fc(V, W_fc[-1])
    y_true_diag = tf.matrix_diag(y_true)
    LLK2 = V @ y_true_diag  # bs x d x K  @  bs x K x K  =  bs x d x K
    l = tf.reduce_sum(LLK2, axis=2)  # bs x d
    l = tf.tile(l, [1, n_out])  # bs x d x K
    l = tf.reshape(l, [-1, n_out, n_in])  # bs x K x d
    V_argmax = tf.transpose(l, perm=[0, 2, 1])  # bs x d x K
    diff_v = tf.abs(V - V_argmax)
    diff_v = diff_v + eps_num_stabil * tf.cast(tf.less(diff_v, eps_num_stabil), tf.float32)
    dist_db_denominator = tf.norm(diff_v, axis=1, ord=q)

    y_pred_diag = tf.expand_dims(y_pred, 1)
    y_pred_correct = y_pred_diag @ y_true_diag  # bs x 1 x K  @  bs x K x K  =  bs x 1 x K
    y_pred_correct = tf.reduce_sum(y_pred_correct, axis=2)  # bs x 1
    y_pred_correct = tf.tile(y_pred_correct, [1, n_out])  # bs x 1 x K
    y_pred_correct = tf.reshape(y_pred_correct, [-1, n_out, 1])  # bs x K x 1
    y_pred_correct = tf.transpose(y_pred_correct, perm=[0, 2, 1])  # bs x 1 x K
    dist_db_numerator = tf.squeeze(y_pred_correct - y_pred_diag, 1)  # bs x K
    dist_db_numerator = dist_db_numerator + 100.0 * y_true  # bs x K

    dist_db = dist_db_numerator / dist_db_denominator + y_true * 2.0 * gamma_db

    th = zero_out_non_min_distances(dist_db, n_db)
    db_term = tf.reduce_sum(th * tf.maximum(0.0, 1.0 - dist_db / gamma_db), axis=1)
    return rb_term, db_term


def mmr_fc(z_list, y_true, W, n_in, n_hs, n_out, n_rb, n_db, gamma_rb, gamma_db, bs, q):
    """
    Batch-wise implementation of the Maximum Margin Regularizer for fully-connected networks.
    Note that it is differentiable, and thus can be directly added to the main objective (e.g. the cross-entropy loss).

    z_list: list with all tensors that correspond to preactivation feature maps
            (in particular, z_list[-1] are logits; see models.MLP for details)
    y_true: one-hot encoded ground truth labels (bs x n_classes)
    W: list with all weight matrices
    n_in: total number of input pixels (e.g. 784 for MNIST)
    n_hs: list of number of hidden units for every hidden layer (e.g. [1024] for FC1)
    n_rb: number of closest region boundaries to take
    n_db: number of closest decision boundaries to take
    gamma_rb: gamma for region boundaries (approx. corresponds to the radius of the Lp-ball that we want to
              certify robustness in)
    gamma_db: gamma for decision boundaries (approx. corresponds to the radius of the Lp-ball that we want to
              be robust in)
    bs: batch size
    q: q-norm which is the dual norm to the p-norm that we aim to be robust at (e.g. if p=np.inf, q=1)
    """
    eps_num_stabil = 1e-5  # epsilon for numerical stability in the denominator of the distances
    n_hl = len(W) - 1  # number of hidden layers
    y_pred = z_list[-1]

    relus = []
    for i in range(n_hl):
        relu = tf.cast(tf.greater(z_list[i], 0), tf.float32)
        relus.append(tf.expand_dims(relu, 1))

    dist_rb = tf.abs(z_list[0]) / tf.norm(W[0], axis=0, ord=q)  # bs x n_hs[0]  due to broadcasting
    V = tf.reshape(tf.tile(W[0], [bs, 1]), [bs, n_in, n_hs[0]])  # bs x d x n1
    V = V * relus[0]  # element-wise mult using broadcasting, result: bs x d x n_cur
    for i in range(1, n_hl):
        V = calc_v_fc(V, W[i])  # bs x d x n_hs[i]
        new_dist_rb = tf.abs(z_list[i]) / tf.norm(V, axis=1, ord=q)  # bs x n_hs[i]
        dist_rb = tf.concat([dist_rb, new_dist_rb], 1)  # bs x sum(n_hs[1:i])
        V = V * relus[i]  # element-wise mult using broadcasting, result: bs x d x n_cur

    th = zero_out_non_min_distances(dist_rb, n_rb)
    rb_term = tf.reduce_sum(th * tf.maximum(0.0, 1.0 - dist_rb / gamma_rb), axis=1)

    # decision boundaries
    V_last = calc_v_fc(V, W[-1])
    y_true_diag = tf.matrix_diag(y_true)
    LLK2 = V_last @ y_true_diag  # bs x d x K  @  bs x K x K  =  bs x d x K
    l = tf.reduce_sum(LLK2, axis=2)  # bs x d
    l = tf.tile(l, [1, n_out])  # bs x d x K
    l = tf.reshape(l, [-1, n_out, n_in])  # bs x K x d
    V_argmax = tf.transpose(l, perm=[0, 2, 1])  # bs x d x K
    diff_v = tf.abs(V_last - V_argmax)
    diff_v = diff_v + eps_num_stabil * tf.cast(tf.less(diff_v, eps_num_stabil), tf.float32)
    dist_db_denominator = tf.norm(diff_v, axis=1, ord=q)

    y_pred_diag = tf.expand_dims(y_pred, 1)
    y_pred_correct = y_pred_diag @ y_true_diag  # bs x 1 x K  @  bs x K x K  =  bs x 1 x K
    y_pred_correct = tf.reduce_sum(y_pred_correct, axis=2)  # bs x 1
    y_pred_correct = tf.tile(y_pred_correct, [1, n_out])  # bs x 1 x K
    y_pred_correct = tf.reshape(y_pred_correct, [-1, n_out, 1])  # bs x K x 1
    y_pred_correct = tf.transpose(y_pred_correct, perm=[0, 2, 1])  # bs x 1 x K
    dist_db_numerator = tf.squeeze(y_pred_correct - y_pred_diag, 1)  # bs x K
    dist_db_numerator = dist_db_numerator + 100.0 * y_true  # bs x K

    dist_db = dist_db_numerator / dist_db_denominator + y_true * 2.0 * gamma_db

    th = zero_out_non_min_distances(dist_db, n_db)
    db_term = tf.reduce_sum(th * tf.maximum(0.0, 1.0 - dist_db / gamma_db), axis=1)
    return rb_term, db_term

