import numpy as np
import tensorflow as tf
from cleverhans.attacks import CarliniWagnerL2, Attack, Model, CallableModelWrapper


def eval_cw_attack(sess, gen_ae_tensor, x_tf, y_tf, logits, is_train, clean_inputs_all, clean_labels_all):
    # Eval CW only on correctly classified points
    logits_val = sess.run(logits, feed_dict={x_tf: clean_inputs_all, is_train: False})
    correctly_classified = np.argmax(clean_labels_all, axis=1) == np.argmax(logits_val, axis=1)
    clean_examples, clean_labels = clean_inputs_all[correctly_classified], clean_labels_all[correctly_classified]

    adv_examples = sess.run(gen_ae_tensor, feed_dict={x_tf: clean_inputs_all, y_tf: clean_labels_all, is_train: False})
    adv_examples = adv_examples[correctly_classified]
    delta_l2 = np.sum((clean_examples - adv_examples) ** 2, axis=(1, 2, 3)) ** .5
    delta_l2_valid_ae = delta_l2[delta_l2 > 0]
    avg_l2_norm = np.mean(delta_l2_valid_ae)

    logits_val_adv = sess.run(logits,
                              feed_dict={x_tf: adv_examples, y_tf: clean_labels, is_train: False})
    correctly_classified_adv = np.argmax(clean_labels, axis=1) == np.argmax(logits_val_adv, axis=1)
    err_rate_adv = 1 - np.mean(correctly_classified_adv)
    return avg_l2_norm, err_rate_adv


def eval_pgd_attack(sess, gen_ae_tensor, x_tf, y_in, logits, is_train, clean_inputs, clean_labels):
    adv_examples = sess.run(gen_ae_tensor, feed_dict={x_tf: clean_inputs, y_in: clean_labels, is_train: False})

    logits_val_adv = sess.run(logits, feed_dict={x_tf: adv_examples, y_in: clean_labels, is_train: False})
    is_ae_misclassified = np.argmax(clean_labels, axis=1) != np.argmax(logits_val_adv, axis=1)
    return is_ae_misclassified


def cw_attack(sess, x, logits, n_ae, final=False):
    cw_attack_obj = CarliniWagnerL2(logits, sess=sess, back='tf')
    if final:
        cw_params = {'binary_search_steps': 9,
                     'max_iterations': 2000,
                     'learning_rate': 0.01,
                     'initial_const': 1.0,
                     'abort_early': True,
                     'batch_size': n_ae
                     }
    else:
        cw_params = {'binary_search_steps': 5,
                     'max_iterations': 500,
                     'learning_rate': 0.01,
                     'initial_const': 1.0,
                     'batch_size': n_ae  # need to specify, since CarliniWagnerL2 is not completely symbolic
                     }
    adv_ex_tensor = cw_attack_obj.generate(x, **cw_params)
    adv_ex_tensor = tf.stop_gradient(adv_ex_tensor)
    return adv_ex_tensor


def pgd_attack(clean_inputs, clean_labels, logits, p_norm, eps, pgd_n_iter):
    """ Symbolic definition of the PGD attack """

    attack = MadryEtAl(logits)
    attack_params = {'nb_iter': pgd_n_iter, 'clip_min': 0.0, 'clip_max': 1.0, 'y': clean_labels, 'ord': p_norm,
                     'eps': eps}
    if p_norm == np.inf:
        attack_params['eps_iter'] = attack_params['eps'] / pgd_n_iter * 2
        attack_params['pgd_update'] = 'sign'
    elif p_norm == 2:
        attack_params['eps_iter'] = attack_params['eps'] / pgd_n_iter * 2
        attack_params['pgd_update'] = 'plain'
    else:
        raise Exception('Wrong p_norm.')

    adv_ex_tensor = attack.generate(clean_inputs, **attack_params)
    adv_ex_tensor = tf.stop_gradient(adv_ex_tensor)
    return adv_ex_tensor


class MadryEtAl(Attack):
    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, sess=None):
        """
        Create a MadryEtAl instance.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(MadryEtAl, self).__init__(model, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, pgd_update='sign', **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init
        self.pgd_update = pgd_update

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        from cleverhans.utils_tf import model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        if self.pgd_update == 'sign':
            adv_x = adv_x + self.eps_iter * tf.sign(grad)
        elif self.pgd_update == 'plain':
            adv_x = adv_x + self.eps_iter * grad / tf.reduce_sum(grad**2, axis=[1, 2, 3], keep_dims=True)**0.5
        else:
            raise Exception('Wrong pgd_update.')
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.
        :param x: A tensor with the input image.
        """
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            if self.ord == np.inf:
                eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
            else:
                eta = tf.random_normal(tf.shape(x), 0.0, 1.0)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
