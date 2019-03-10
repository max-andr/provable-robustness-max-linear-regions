import argparse

import numpy as np
import os
import scipy.io
import torch
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

from kolter_wong.convex_adversarial import epsilon_from_model


def data_loader(dataset, batch_size, shuffle_test=False):

    if dataset == 'mnist':
        train_data = datasets.MNIST("./data/mnist", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST("./data/mnist", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'fmnist':
        train_data = datasets.FashionMNIST("./data/fmnist", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.FashionMNIST("./data/fmnist", train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10("./data/cifar10", train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, 4),
                                          transforms.ToTensor(),
                                      ]))
        test_data = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'gts':
        train = scipy.io.loadmat('datasets/{}/{}_int_train.mat'.format(dataset, dataset))
        test = scipy.io.loadmat('datasets/{}/{}_int_train.mat'.format(dataset, dataset))
        x_train, y_train, x_test, y_test = train['images'], train['labels'], test['images'], test['labels']

        X_te = torch.from_numpy(x_test).float().permute([0, 3, 1, 2])  # NHWC to NCHW
        X_tr = torch.from_numpy(x_train).float().permute([0, 3, 1, 2])  # NHWC to NCHW
        y_te = torch.from_numpy(y_test).long()
        y_tr = torch.from_numpy(y_train).long()

        train_data = td.TensorDataset(X_tr, y_tr)
        test_data = td.TensorDataset(X_te, y_te)
    else:
        raise ValueError('wrong dataset')

    pin_memory = True
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_test, pin_memory=pin_memory)
    return train_loader, test_loader


def fashion_mnist_loaders(batch_size):
    mnist_train = datasets.MNIST("./fashion_mnist", train=True,
                                 download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./fashion_mnist", train=False,
                                download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('./data', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def argparser(batch_size=32, epochs=20, seed=0, verbose=0, lr=1e-3,
              epsilon=0.1, starting_epsilon=None,
              l1_proj=None, delta=None, m=1, l1_eps=None,
              l1_train='exact', l1_test='exact',
              opt='adam', momentum=0.9, weight_decay=5e-4):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mnist')

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--p", type=str)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=10)

    # projection settings
    parser.add_argument('--l1_proj', type=int, default=l1_proj)
    parser.add_argument('--delta', type=float, default=delta)
    parser.add_argument('--m', type=int, default=m)
    parser.add_argument('--l1_train', default=l1_train)
    parser.add_argument('--l1_test', default=l1_test)
    parser.add_argument('--l1_eps', type=float, default=l1_eps)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix:
        if args.model is not None:
            args.prefix += '_' + args.model

        if args.method is not None:
            args.prefix += '_' + args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval',
                  'method', 'model', 'cuda_ids', 'load']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length',
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # if not using adam, ignore momentum and weight decay
        if args.opt == 'adam':
            banned += ['momentum', 'weight_decay']

        if args.m == 1:
            banned += ['m']
        if args.cascade == 1:
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']:
            banned += ['model_factor']

        if args.model != 'resnet':
            banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)):
            if arg not in banned and getattr(args, arg) is not None:
                args.prefix += '_' + arg + '_' + str(getattr(args, arg))

        # if args.schedule_length > args.epochs:
        #     raise ValueError('Schedule length for epsilon ({}) is greater than '
        #                      'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else:
        args.prefix = 'temporary'

    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    return args


def args2kwargs(args, X=None):
    if args.l1_proj is not None:
        if not args.l1_eps:
            if args.delta:
                args.l1_eps = epsilon_from_model(model, Variable(X.cuda()), args.l1_proj,
                                                 args.delta, args.m)
                print('''
        With probability {} and projection into {} dimensions and a max
        over {} estimates, we have epsilon={}'''.format(args.delta, args.l1_proj,
                                                        args.m, args.l1_eps))
            else:
                args.l1_eps = 0
                print('No epsilon or \delta specified, using epsilon=0.')
        else:
            print('Specified l1_epsilon={}'.format(args.l1_eps))
        kwargs = {
            'l1_proj': args.l1_proj,
            # 'l1_eps' : args.l1_eps,
            # 'm' : args.m
        }
    else:
        kwargs = {
        }
    return kwargs
