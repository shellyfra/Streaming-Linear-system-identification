import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit
import time
from copy import deepcopy
from algorithms_v2 import SGD
from algorithms_v2 import SGD_ER
from algorithms_v2 import SGD_RER
from algorithms_v2 import SGD_MLMC
from envs.simple_chain import SimpleChainMNISTLogisticRegression, SimpleChainLinearRegression
from envs.linear_system import *
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, \
    set_default_plot_params, save_fig
from utils.plot_utils import plot_convergence
from utils.projection_utils import project_l2_ball
from utils.learning_utils import create_averages_array, compute_OLS_error_arr
from scipy import interpolate
from scipy.linalg import eigh
from utils.plot_utils import plot_all_dict_instances
from utils.plot_utils import plot_instance

set_default_plot_params()

# Option to load/save
load_run = False  # False/True If true just load results from dir, o.w run simulations
result_dir_to_load = 'results\\RandBiMod_pred_error\\SGD'
save_result = True

# Plot options
save_PDF = False  # False/True - save figures as PDF file in the result folder
plot = True
plot_from_mem = False  # plot all the data that we have in results folder
# -------------------------------------------------------------------------------------------
# Set Parameters
# -------------------------------------------------------------------------------------------
args = argparse.Namespace()

# ----- Run Parameters ---------------------------------------------#
args.exp = 'RandBiMod'  # '____' | '_____' | '____'
if args.exp == 'RandBiMod':
    args.d = 5  # A* matrix size
    args.rho = 0.9  # values in the eigenvalue are rho and rho/3
    args.sigma = 0.1  # noise distribution
    args.T = 100_000  # horizon
    args.X0 = None
    args.result_dir = 'RandBiMod_pred_error'
    args.simple_A_star = False  # True -> A = rho*I
    # args.gap_size = 10
    # args.buffer_size = 10*args.gap_size
else:
    raise NotImplementedError

# args.run_name = 'SGD_001_over_sqrtT'  # 'Name of dir to save results in (if empty, name by time)'
args.run_name = 'SGD'
args.seed = 20  # Random seed
args.n_reps = 3  # Number of experiment repetitions. Default: LinearRegression - 10, MNIST - 20
args.evaluate_on_average = True
args.subsample = 100  # for plotting
args.calc_prediction_error = True

# args.optimizer = 'SGD'  # 'SGD' | 'SGD_MLMC' | 'SGD_DD' | 'SGD_ER' | 'SGD_RER'
args.optimizer = 'SGD'
args.lr_params = {'type': '1/2R', 'n_samples_for_estimating_R': int(np.floor(2 * np.log(args.T)))}  # From paper SGD-ER
# args.lr_params = {'type': 'AdaGrad', 'alpha': 1}
if args.optimizer == 'SGD_RER' or args.optimizer == 'SGD_ER':
    args.B = 100
    args.lr_params["B"] = args.B
    args.u = 10  # gap size
# args.lr_params = {'type': 'alpha/sqrt(t)', 'alpha': 1}  # Optimally tuned: 0.01
if args.optimizer == "SGD_MLMC":
    args.grad_clip = None  # Element-wise gradient clip: float | None
    # truncated geom. dist.; None: standard geom. dist. Default: LinearRegression - 5, MNIST - 7.
    args.truncate_geometric_at = 7
    args.momentum_def = None  # None | 'standard' | 'corrected'
    args.beta = 0.9  # Momentum parameter

# args.lr_params = {'type': '1/8RB', 'n_samples_for_estimating_R': int(np.floor(2 * np.log(args.T)))}  # From paper SGD-RER
# args.lr_params = {'type': 'const', 'alpha': 0.001}
# args.lr_params = {'alpha': 1}
# args.momentum_def = None  # None | 'standard' | 'corrected'
# args.grad_clip = None  # Element-wise gradient clip: float | None
# args.batch_size = None  # batch size for SGD; None/1 is standard SGD.
# # -------------------------------------------------------------------------------------------


def single_simulation(args):
    project = False
    if args.exp == 'RandBiMod':
        # X0 = 2*np.random.randn(args.d).reshape(-1, 1)
        objective = RandBiMod(d=args.d, rho=args.rho, sigma=args.sigma, X0=args.X0, simple_A_star=args.simple_A_star)
    else:
        raise NotImplementedError
    # Draw initial guess
    A_init = np.zeros((args.d, args.d))  # from paper
    # A_init = 0.01 * np.random.randn(args.d, args.d)
    # Estimate the norm of the iterates by running the process
    if (args.lr_params['type'] == '1/2R') or (args.lr_params['type'] == '1/8RB'):
        R = 0
        for _ in range(args.lr_params['n_samples_for_estimating_R']):
            R += np.linalg.norm(objective.get_curr_x())
            objective.step()
        args.lr_params['R'] = R
    else:
        R = None
        # args.T = args.T - args.lr_params['n_samples_for_estimating_R']  # Ignore samples used for estimation
    # Set optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(w_init=A_init, lr_params=args.lr_params, momentum_def=None,
                        beta=None, grad_clip=None)
    elif args.optimizer == 'SGD_ER':
        optimizer = SGD_ER(w_init=A_init, lr_params=args.lr_params,  buff_size=args.B, buffer_gap=args.u, momentum_def=None,
                           beta=None, grad_clip=None)
    elif args.optimizer == 'SGD_RER':
        optimizer = SGD_RER(w_init=A_init, lr_params=args.lr_params, buff_size=args.B, buffer_gap=args.u, R=R, momentum_def=None,
                            beta=None, grad_clip=None)
    elif args.optimizer == 'SGD_MLMC':
        # if args.encode_mix_time_in_lr and args.epsilon is not None:
        #     args.lr_params['alpha'] = np.sqrt(args.epsilon)
        #     if args.lr_params['AdaGrad']:
        #         print('Encoding mixing time in AdaGrad..')
        optimizer = SGD_MLMC(w_init=A_init, lr_params=args.lr_params, momentum_def=args.momentum_def,
                             beta=args.beta, grad_clip=args.grad_clip, truncate_at=args.truncate_geometric_at)
    else:
        raise NotImplementedError
    # Run T steps
    optimizer.run(T=args.T, objective=objective, project=project)

    # Compute loss
    iterates = np.array(optimizer.iterates)
    if args.evaluate_on_average:
        iterates = create_averages_array(iterates)  # TODO: Add averaging starting a given time

    if args.optimizer == 'SGD_MLMC':
        observed_samples_axis = np.cumsum(np.array([optimizer.n_samples_per_step_list]))
    else:
        observed_samples_axis = None

    # error = np.linalg.norm(iterates - objective.A_star, ord=2, axis=(1, 2)) # todo add to evaluate func for pred / parameter error that
    error = objective.evaluate_error(iterates, args.calc_prediction_error)
    # will be a param
    if args.optimizer == "SGD" and 'n_samples_for_estimating_R' not in args.lr_params.keys():
        objective.step()

    trajectory = np.hstack(objective.trajectory)
    A_star = objective.A_star
    return error, trajectory, A_star, observed_samples_axis


def run_simulations(arguments, save_result):
    if save_result:
        create_result_dir(arguments)
        write_to_log('Creating results directory', arguments)

    error_mat, OLS_err_mat = [], []
    start_time = time.time()
    # ----- Run simulations ------ #
    for i_rep in range(arguments.n_reps):
        set_random_seed(arguments.seed + i_rep)
        error, trajectory, A_star, observed_samples_axis = single_simulation(arguments)
        if 'n_samples_for_estimating_R' in arguments.lr_params.keys():
            start_from = arguments.lr_params['n_samples_for_estimating_R']
        else:
            start_from = 2

        if args.optimizer == 'SGD':
            OLS_err = compute_OLS_error_arr(trajectory, A_star, start_from=start_from, subsample=arguments.subsample)
            OLS_err_mat.append(OLS_err)
        elif args.optimizer == "SGD_MLMC":
            interpolated_loss = interpolate.interp1d(observed_samples_axis, error, kind='cubic')
            new_axis = np.array(np.arange(args.T + 1))
            error = interpolated_loss(new_axis)

        error_mat.append(error)
        print('Rep {} is done! Elapsed time: {:.3f}[s]'.format(i_rep + 1, time.time() - start_time))
        start_time = time.time()

    error_mat = np.vstack(error_mat)
    if args.optimizer == 'SGD':
        OLS_err_mat = np.vstack(OLS_err_mat)
        info_dict = {'error_avg': error_mat.mean(axis=0),
                     'error_std': error_mat.std(axis=0),
                     'OLS_error_avg': OLS_err_mat.mean(axis=0),
                     'OLS_error_std': OLS_err_mat.std(axis=0)}
    else:
        info_dict = {'error_avg': error_mat.mean(axis=0),
                     'error_std': error_mat.std(axis=0)}
    if save_result:
        save_run_data(arguments, info_dict)
    return info_dict


def main(args, save_result=True, load_run_data_flag=False, result_dir_to_load='',
         save_PDF=False, plot=True, plot_from_mem=False):
    if load_run_data_flag:
        # args, disp_errors, disp_errors_logT = load_run_data(result_dir_to_load)
        args, info_dict = load_run_data(result_dir_to_load)
        plot_instance(args, info_dict, plot, save_PDF)
    elif plot_from_mem:
        plot_all_dict_instances()
    else:
        info_dict = run_simulations(args, save_result)
        plot_instance(args, info_dict, plot, save_PDF)


if __name__ == '__main__':
    main(args, save_result=save_result, load_run_data_flag=load_run,
         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, plot=plot, plot_from_mem=plot_from_mem)
    print('Done!')
