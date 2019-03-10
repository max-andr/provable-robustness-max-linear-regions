import numpy as np
import scipy.io

import torch
import torch.nn as nn
import math
import data

from kolter_wong.convex_adversarial import Dense, DenseSequential
from kolter_wong.custom_layers import Conv2dUntiedBias


def select_model(model_type, n_in, n_out):
    h_in, w_in, c_in = (28, 28, 1) if n_in == 28*28*1 else (32, 32, 3)
    if 'fc' in model_type:
        n_h_layers = int(model_type.split('fc')[-1])
        if model_type == 'fc10':  # manual hack to have the same model as we reported
            n_hs = [124, 104, 104, 104, 104, 104, 104, 104, 86, 86]
        else:
            n_hs = n_h_layers * [1024]
        n_hs = [n_in] + n_hs + [n_out]
        model = fc(n_hs)
    elif model_type == 'cnn_lenet_avgpool':
        model = lenet_avgpool(h_in, w_in, c_in, n_out)
    elif model_type == 'cnn_lenet_small':
        model = lenet_small(h_in, w_in, c_in, n_out)
    elif model_type == 'cnn_lenet_large':
        model = lenet_large(h_in, w_in, c_in, n_out)
    else:
        raise ValueError('wrong model_type')
    return model


def fc(n_hs):
    layers = [Flatten()]
    for i in range(len(n_hs) - 2):
        layers.append(nn.Linear(n_hs[i], n_hs[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(n_hs[-2], n_hs[-1]))

    model = nn.Sequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def lenet_avgpool(h_in, w_in, c_in, n_out):
    model = nn.Sequential(
        Conv2dUntiedBias(24, 24, c_in, 16, 5, stride=1, padding=0),
        # nn.Conv2d(1, 16, 5, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 16, 2, stride=2, padding=0, bias=None),  # aka nn.AvgPool2d(2, stride=2),
        Conv2dUntiedBias(8, 8, 16, 32, 5, stride=1, padding=0),
        # nn.Conv2d(16, 32, 5, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 32, 2, stride=2, padding=0, bias=None),  # aka nn.AvgPool2d(2, stride=2),
        Flatten(),
        nn.Linear(4 * 4 * 32, n_out)
    )

    # Proper default init (not needed if we just evaluate with KW code)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def lenet_small(h_in, w_in, c_in, n_out):
    model = nn.Sequential(
        nn.Conv2d(c_in, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * h_in//4 * w_in//4, 100),
        nn.ReLU(),
        nn.Linear(100, n_out)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def lenet_large(h_in, w_in, c_in, n_out):
    model = nn.Sequential(
        nn.Conv2d(c_in, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64 * h_in//4 * w_in//4, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_out)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def resnet(N=5, factor=10):
    """
    Original CIFAR-10 ResNet proposed in He et al.
    :param N:
    :param factor:
    :return:
    """

    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)),
            nn.ReLU(),
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0),
                  None,
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)),
            nn.ReLU()
        ]

    conv1 = [nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU()]
    conv2 = block(16, 16 * factor, 3, False)
    for _ in range(N):
        conv2.extend(block(16 * factor, 16 * factor, 3, False))
    conv3 = block(16 * factor, 32 * factor, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(32 * factor, 32 * factor, 3, False))
    conv4 = block(32 * factor, 64 * factor, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(64 * factor, 64 * factor, 3, False))
    layers = (
            conv1 +
            conv2 +
            conv3 +
            conv4 +
            [Flatten(),
             nn.Linear(64 * factor * 8 * 8, 1000),
             nn.ReLU(),
             nn.Linear(1000, 10)]
    )
    model = DenseSequential(
        *layers
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def restore_model(sess, model_pt, model_tf, nn_type, device):
    vars_pt = list(model_pt.parameters())
    Ws, bs = model_tf.W, model_tf.b

    vars_tf = []
    for W, b in zip(Ws, bs):
        vars_tf.append(W)
        vars_tf.append(b)
    assert len(vars_pt) == len(vars_tf)

    for var_pt, var_tf in zip(vars_pt, vars_tf):
        var_np = sess.run(var_tf)

        if 'weights_conv' in var_tf.name:
            var_np = np.transpose(var_np, [3, 2, 0, 1])
        elif 'weights_fc1' in var_tf.name:
            n_in, n_out = var_np.shape
            h = w = int(math.sqrt(var_np.shape[0] / model_tf.n_filters[-1]))
            var_np = np.transpose(var_np)
            var_np = var_np.reshape([n_out, h, w, model_tf.n_filters[-1]])
            var_np = var_np.transpose([0, 3, 1, 2])
            var_np = var_np.reshape([n_out, n_in])
        elif 'weight' in var_tf.name:
            var_np = np.transpose(var_np)
        elif 'bias' in var_tf.name:
            var_np = var_np.flatten()  # needed only for FC

        var_pt.data = torch.from_numpy(var_np).to(device)

