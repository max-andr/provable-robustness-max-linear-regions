import argparse
import sys
import os
import scipy.io

sys.path.append('..')  # to import data.py from the parent directory
sys.path.append('mip')
import utils
import parse


def convert_mat_tf_to_mip(mat_tf, nn_type):
    """ Direct convertation from our TF format to the format needed by MIP """

    if nn_type == 'fc1':
        weight_names_tf = ['U', 'W']
        bias_names_tf = ['bU', 'bW']
        var_names_mip = ['fc1', 'softmax']
    elif nn_type == 'cnn_lenet_small':
        weight_names_tf = ['weights_conv1', 'weights_conv2', 'weights_fc1', 'weights_fc2']
        bias_names_tf = ['biases_conv1', 'biases_conv2', 'biases_fc1', 'biases_fc2']
        var_names_mip = ['conv1', 'conv2', 'fc1', 'logits']
    else:
        raise ValueError('wrong nn_type')
    weight_names_julia = [wn + '/weight' for wn in var_names_mip]
    bias_names_julia = [wn + '/bias' for wn in var_names_mip]

    mat_mip = {}
    for var_name_tf, var_name_mip in zip(weight_names_tf + bias_names_tf, weight_names_julia + bias_names_julia):
        var_np = mat_tf[var_name_tf]
        if 'fc' in nn_type:  # for FC models only
            if var_name_tf[0] == 'b' or 'bias' in var_name_tf:
                var_np = var_np.flatten()  # needed only for FC
        elif 'cnn' in nn_type:
            pass
        else:
            raise ValueError('wrong nn_type')
        mat_mip[var_name_mip] = var_np

    return mat_mip


def verify(mat_tf_file, dataset, nn_type, norm_str, eps, n_eval, time_limit, presolver, time_limit_bounds=5):
    # Example of `mat_tf_file`: model.mat
    mat_mip_file = mat_tf_file.replace('.mat', '_mip.mat')  # now: model_mip.mat
    out_mip_file_dir = mat_tf_file.replace('.mat', ' eps={}/'.format(eps))

    # Convertation of the .mat file to the format supported by MIPVerify.jl
    mat_tf = scipy.io.loadmat(mat_tf_file)
    mat_mip = convert_mat_tf_to_mip(mat_tf, nn_type)
    utils.create_folders([out_mip_file_dir])
    scipy.io.savemat(mat_mip_file, mat_mip)

    julia_command_to_run = 'julia mip/verify.jl "{}" "{}" {} {} {} {} {} {} {} {}'.format(
        mat_mip_file, out_mip_file_dir, dataset, nn_type, norm_str, eps, n_eval, time_limit,
        time_limit_bounds, presolver)
    print(julia_command_to_run)

    os.system(julia_command_to_run)


def parse_summary(mat_tf_file, dataset, eps):
    out_mip_file_dir = mat_tf_file.replace('.mat', ' eps={}/'.format(eps))
    out_mip_file = out_mip_file_dir + 'summary.csv'

    summary = parse.get_summary(out_mip_file, dataset)
    error_lb, error_ub = float(summary['Robust Error, LB']), float(summary['Robust Error, UB'])
    error_clean = float(summary['RegularError'])
    frac_timeout = float(summary['TimeoutNumber'])
    total_time = float(summary["TotalTime"])
    n_points = int(summary["Count"])
    return error_clean, error_lb, error_ub, frac_timeout, total_time, n_points


def parse_bounds_pointwise(mat_tf_file, dataset, eps):
    out_mip_file_dir = mat_tf_file.replace('.mat', ' eps={}/'.format(eps))
    out_mip_file = out_mip_file_dir + 'summary.csv'

    is_non_robust, is_provably_robust = parse.get_bounds_pointwise(out_mip_file, dataset)
    return is_non_robust, is_provably_robust


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='Whether to re-eval MIP or just output precalculated summaries.')
    parser.add_argument('--nn_type', type=str, default='cnn_lenet_small', help='NN architecture: fc1, cnn_lenet_small')
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist, fmnist, gts, cifar10')
    parser.add_argument('--norm_str', type=str, default='inf', help='inf, 2')
    parser.add_argument('--eps', type=float, default=0.1, help='Linf epsilon robustness threshold')
    parser.add_argument('--n_eval', type=int, default=1000, help='Number of first test points to evaluate on.')
    parser.add_argument('--time_limit', type=int, default=120, help='Time limit of MIP solver (in sec).')
    parser.add_argument('--presolver', type=str, default='lp', help='ia or lp')
    parser.add_argument('--mat_tf_file', type=str, help='Filename of matlab file in my format')

    args = parser.parse_args()

    if args.eval:
        verify(args.mat_tf_file, args.dataset, args.nn_type, args.norm_str, args.eps, args.n_eval, args.time_limit,
               args.presolver)
    else:
        error_clean, error_lb, error_ub, frac_timeout, total_time, n_points = parse_summary(args.mat_tf_file,
                                                                                            args.dataset, args.eps)
        model_name_short = args.mat_tf_file.split('/')[-1].replace('.mat', '')
        print(model_name_short)
        print(
            'Error clean: {:.2%}, error lb: {:.2%}, error ub: {:.2%}, frac timeout {:.2%} ({} pts in {:.2f} min)'.format(
                error_clean, error_lb, error_ub, frac_timeout, n_points, total_time / 60))
