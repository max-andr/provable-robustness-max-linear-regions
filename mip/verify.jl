#=
MIP verification of neural networks using MIPVerify.jl library.
Note: L1, L2 and Linf are supported.

Example on how to run:
`julia verify.jl "model.mat" "model eps=0.1/" mnist cnn_lenet_small inf 0.1 1000 120 5 lp`
- `model.mat`: the name of the .mat file containing model weights (note, mip_verify.py contains a
script on how to transform the weights from the original format to the one supported by MIPVerify.jl)
- `model eps=0.1/`: the name of the folder that will contain the output of the verifier
- `mnist`: the dataset
- `cnn_lenet_small`: the CNN architecture (the definition is given below)
- `inf`: the Lp-norm used to verify the existence of adversarial examples
- `0.1`: the radius of the Lp-norm
- `1000`: the number of points to verify
- `120`: the timeout of the verifier (in seconds)
- `5`: the timeout of the presolver (in seconds)
- `lp`: the type of the presolver (lp - linear programming, ia - interval arithmetics)
=#

using MIPVerify
using Gurobi
using Memento
using MAT


mat_mip_file = ARGS[1]
out_mip_file_dir = ARGS[2]
dataset_name = ARGS[3]
nn_type = ARGS[4]
norm_str = ARGS[5]
eps = parse(Float64, ARGS[6])
n_eval = parse(Int, ARGS[7])
time_limit = parse(Int, ARGS[8])
time_limit_bounds = parse(Int, ARGS[9])
presolver = ARGS[10]

if norm_str == "inf"
	norm = Inf
elseif norm_str == "2"
	norm = 2
else
	error("wrong norm_str")
end

if presolver == "ia"
	tightening_alg = interval_arithmetic
elseif presolver == "lp"
	tightening_alg = lp
else
	error("wrong presolver")
end

param_dict = mat_mip_file |> matread;
dataset = read_datasets(dataset_name)

n_h = size(dataset.test.images, 2)
n_w = size(dataset.test.images, 3)
n_c = size(dataset.test.images, 4)
n_out = length(unique(dataset.test.labels))
n_in = n_h * n_w * n_c  # needed only for FC networks
if nn_type == "fc1"
	n_hidden = 1024
	fc1 = get_matrix_params(param_dict, "fc1", (n_in, n_hidden))
	softmax = get_matrix_params(param_dict, "softmax", (n_hidden, n_out))

	nnparams = Sequential([
		Flatten(4),
		fc1, ReLU(interval_arithmetic),
		softmax
		],
		"$(out_mip_file_dir)"
	)
elseif nn_type == "cnn_lenet_small"
	conv1 = get_conv_params(param_dict, "conv1", (4, 4, n_c, 16), expected_stride = 2)

	conv2 = get_conv_params(param_dict, "conv2", (4, 4, 16, 32), expected_stride = 2)

	fc1 = get_matrix_params(param_dict, "fc1", (div(div(n_h, 2), 2)^2 * 32, 100))
	softmax = get_matrix_params(param_dict, "logits", (100, n_out))

	nnparams = Sequential(
		[   conv1, ReLU(interval_arithmetic),
		conv2, ReLU(),
# 		Flatten([1, 3, 2, 4]),  # this works for PyTorch standard
		Flatten([1, 4, 3, 2]),  # this works for TF standard
		fc1, ReLU(),
		softmax],
		"$(out_mip_file_dir)"
	)
else
	error("wrong model_name")
end

# Important to check whether we imported the model correctly
f = frac_correct(nnparams, dataset.test, 100)
println("Fraction correct of the first 100: $(f)")

# On which points to run MIP
target_indexes = 1:n_eval

MIPVerify.setloglevel!("info")  # "info" to get the number of unstable ReLUs

if norm == Inf
	MIPVerify.batch_find_untargeted_attack(
		nnparams,
		dataset.test,
		target_indexes,
		GurobiSolver(Gurobi.Env(), BestObjStop=eps, TimeLimit=time_limit),
		norm_order=Inf,
		tightening_algorithm=tightening_alg,
		rebuild=true,
		cache_model=false,
		solve_if_predicted_in_targeted=false,
		tightening_solver=GurobiSolver(Gurobi.Env(), TimeLimit=time_limit_bounds, OutputFlag=0),
	    pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps),
		solve_rerun_option = MIPVerify.never
	)
elseif norm == 2
	MIPVerify.batch_find_untargeted_attack(
		nnparams,
		dataset.test,
		target_indexes,
		GurobiSolver(Gurobi.Env(), BestObjStop=eps, TimeLimit=time_limit),
		norm_order=2,
		tightening_algorithm=tightening_alg,
		rebuild=true,
		cache_model=false,
		solve_if_predicted_in_targeted=false,
		tightening_solver=GurobiSolver(Gurobi.Env(), TimeLimit=time_limit_bounds, OutputFlag=0),
	    pp = MIPVerify.L2NormBoundedPerturbationFamily(eps),
		solve_rerun_option = MIPVerify.never
	)
else
	error("wrong norm of perturbations")
end

