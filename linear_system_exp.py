import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit
import time
from copy import deepcopy
from algorithms_v2 import SGD
from algorithms_v2 import SGD_ER
from envs.simple_chain import SimpleChainMNISTLogisticRegression, SimpleChainLinearRegression
from envs.linear_system import *
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, \
    set_default_plot_params, save_fig
from utils.plot_utils import plot_convergence
from utils.projection_utils import project_l2_ball
from utils.learning_utils import create_averages_array, compute_OLS_error_arr
from scipy import interpolate
from scipy.linalg import eigh

set_default_plot_params()

# Option to load/save
load_run = False  # False/True If true just load results from dir, o.w run simulations
result_dir_to_load = 'results\\SGD_and_OLS\\LinearSystem'
save_result = True

# Plot options
save_PDF = False  # False/True - save figures as PDF file in the result folder
plot = True

# -------------------------------------------------------------------------------------------
# Set Parameters
# -------------------------------------------------------------------------------------------
args = argparse.Namespace()

# ----- Run Parameters ---------------------------------------------#
args.exp = 'RandBiMod'  # '____' | '_____' | '____'
if args.exp == 'RandBiMod':
    args.d = 5  # A* matrix size
    args.rho = 0.9  # values in the the eigenvalue are rho and rho/3
    args.sigma = 1  # noise distribution
    args.T = 1000000  # horizon
    args.X0 = None
    args.result_dir = 'RandBiMod'
    # args.gap_size = 10
    # args.buffer_size = 10*args.gap_size
else:
    raise NotImplementedError

args.run_name = 'SGD_001_over_sqrtT'  # 'Name of dir to save results in (if empty, name by time)'
args.seed = 41  # Random seed
args.n_reps = 3  # Number of experiment repetitions. Default: LinearRegression - 10, MNIST - 20
args.evaluate_on_average = True
args.subsample = 1000  # for plotting

# args.optimizer = 'SGD'  # 'SGD' | 'SGD_MLMC' | 'SGD_DD' | 'SGD_ER'
args.optimizer = 'SGD'
args.lr_params = {'type': '1/2R', 'n_samples_for_estimating_R': int(np.floor(2 * np.log(args.T)))}  # From paper


# args.lr_params = {'type': 'alpha/sqrt(t)', 'alpha': 1}  # Optimally tuned: 0.01
# args.lr_params = {'type': 'AdaGrad', 'alpha': 1}
# args.lr_params = {'type': 'const', 'alpha': 0.001}
# args.lr_params = {'alpha': 1}
# args.momentum_def = None  # None | 'standard' | 'corrected'
# args.beta = 0.9  # Momentum parameter
# args.grad_clip = None  # Element-wise gradient clip: float | None
# args.batch_size = None  # batch size for SGD; None/1 is standard SGD.
# # -------------------------------------------------------------------------------------------


def single_simulation(args):
    project = False
    if args.exp == 'RandBiMod':
        # X0 = 2*np.random.randn(args.d).reshape(-1, 1)
        objective = RandBiMod(d=args.d, rho=args.rho, sigma=args.sigma, X0=args.X0)
    else:
        raise NotImplementedError
    # Draw initial guess
    A_init = 0.01 * np.random.randn(args.d, args.d)
    # Estimate the norm of the iterates by running the process
    if args.lr_params['type'] == '1/2R':
        R = 0
        for _ in range(args.lr_params['n_samples_for_estimating_R']):
            R += np.linalg.norm(objective.get_curr_x())
            objective.step()
        args.lr_params['R'] = R
        # args.T = args.T - args.lr_params['n_samples_for_estimating_R']  # Ignore samples used for estimation
    # Set optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(w_init=A_init, lr_params=args.lr_params, momentum_def=None,
                        beta=None, grad_clip=None)
    elif args.optimizer == 'SGD_ER':
        optimizer = SGD_ER(w_init=A_init, lr_params=args.lr_params, momentum_def=None,
                           beta=None, grad_clip=None)
    else:
        raise NotImplementedError
    # Run T steps
    optimizer.run(T=args.T, objective=objective, project=project)

    # Compute loss
    iterates = np.array(optimizer.iterates)
    if args.evaluate_on_average:
        iterates = create_averages_array(iterates)  # TODO: Add averaging starting a given time

    error = np.linalg.norm(iterates - objective.A_star, ord=2, axis=(1, 2))
    trajectory = np.hstack(objective.trajectory)
    A_star = objective.A_star

    #     # observed_samples_axis = np.arange(args.T + 1)  # 1 sample per step
    #     print("first point = ", points[0][0], " ", points[1][0])
    #     x, y = np.array(points[0]), np.array(points[1])
    # #    colors = np.random.randint(x.size, size=x.size)
    #     colors = np.arange(x.size)
    #     colors = colors.sort()
    #     plt.scatter(x, y, c=colors) #, cmap="blues"
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.xlim(-5, 5)
    #     plt.ylim(-5, 5)
    #     plt.colorbar()
    #     plt.tight_layout()
    #     plt.show()
    # plt.plot(observed_samples_axis, points, 'o', color='black')
    return error, trajectory, A_star


# def compute_param_error(A_star, X_points, A_matrix_array, T):
#     # add X_points for future usage
#     # compute A moving avg
#     num_of_iterations = X_points.shape[1]  # without first 2logT for the lr
#     errors = []
#     errors_logT = []
#     logT = int(np.round(np.log(T)))
#     for i in range(num_of_iterations):
#         test = np.stack(A_matrix_array[:i + 1])
#         A_moving_avg = test.mean(axis=0)
#         err_val = A_moving_avg - A_star
#         lambda_max = eigh(err_val, eigvals_only=True)[-1]
#         A_norm = np.linalg.norm(err_val, 'fro')
#         errors.append(A_norm)
#
#         if i >= logT:
#             A_moving_avg_logT = A_matrix_array[logT:i + 1].mean(
#                 axis=0)  # todo: check ! "maintain a running tail average at the end of each of the subsequent buffers"
#             err_val_logT = A_moving_avg_logT - A_star
#             lambda_max_logT = eigh(err_val_logT, eigvals_only=True)
#             # A_norm_logT = np.sqrt(lambda_max_logT * (np.matmul(err_val_logT, err_val_logT.T)))
#             A_norm_logT = np.linalg.norm(err_val, 'fro')
#             errors_logT.append(A_norm_logT)
#     return errors, errors_logT


def run_simulations(arguments, save_result):
    if save_result:
        create_result_dir(arguments)
        write_to_log('Creating results directory', arguments)

    error_mat, OLS_err_mat = [], []
    start_time = time.time()
    # ----- Run simulations ------ #
    for i_rep in range(arguments.n_reps):
        set_random_seed(arguments.seed + i_rep)
        error, trajectory, A_star = single_simulation(arguments)
        if 'n_samples_for_estimating_R' in arguments.lr_params.keys():
            start_from = arguments.lr_params['n_samples_for_estimating_R']
        else:
            start_from = 2
        error_mat.append(error)
        OLS_err = compute_OLS_error_arr(trajectory, A_star, start_from=start_from, subsample=arguments.subsample)
        OLS_err_mat.append(OLS_err)
        print('Rep {} is done! Elapsed time: {:.3f}[s]'.format(i_rep + 1, time.time() - start_time))

    # disp_errors, disp_errors_logT = compute_param_error(optimal_A, X_points, A_matrix_array, arguments.T)
    # return np.array(disp_errors).reshape(-1, 1), np.array(disp_errors_logT).reshape(-1, 1)

    error_mat = np.vstack(error_mat)
    OLS_err_mat = np.vstack(OLS_err_mat)
    info_dict = {'error_avg': error_mat.mean(axis=0),
                 'error_std': error_mat.std(axis=0),
                 'OLS_error_avg': OLS_err_mat.mean(axis=0),
                 'OLS_error_std': OLS_err_mat.std(axis=0)}
    if save_result:
        save_run_data(args, info_dict)
    return info_dict


def main(args, save_result=True, load_run_data_flag=False, result_dir_to_load='',
         save_PDF=False, plot=True):
    if load_run_data_flag:
        # args, disp_errors, disp_errors_logT = load_run_data(result_dir_to_load)
        args, info_dict = load_run_data(result_dir_to_load)
    else:
        # disp_errors, disp_errors_logT = run_simulations(args, save_result)
        info_dict = run_simulations(args, save_result)
        # _, info_dict2 = load_run_data(result_dir_to_load)
        # info_dict.update(info_dict2)

    error_avg, error_std = info_dict["error_avg"], info_dict["error_std"]
    OLS_error_avg, OLS_error_std = info_dict["OLS_error_avg"], info_dict["OLS_error_std"]
    # SGD_ER_errorr_avg, SGD_ER_errorr_std = info_dict["SGD_ER_error_avg"], info_dict["SGD_ER_error_std"]
    # ----- Plot figures ----- #
    if plot or save_PDF:
        plot_convergence(error_avg, error_std, n_reps=args.n_reps, subsample=args.subsample, label=args.run_name)
        plot_convergence(OLS_error_avg, OLS_error_std, xaxis=np.arange(1, error_avg.shape[0] + 1, args.subsample),
                         subsample=1, n_reps=args.n_reps, label='OLS')
        # x = np.arange(1, disp_errors.shape[0]+1)
        # plt.scatter(x, disp_errors.reshape(-1))
        plt.xlim(1, args.T)
        plt.xlabel('Num. iterations')
        plt.ylabel(r'$||\overline{A}_{t} - A^*||$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        if save_PDF:
            save_fig(args.run_name + '_error')
        else:
            plt.show()
        # plot_convergence(disp_errors_logT, n_reps=args.T - int(np.floor(np.log2(args.T))), subsample=10) # semilogx=True ??
        # plt.xlabel('Num. iterations')
        # plt.ylabel(r'$||\bar{A}_{log(T)t} - A^*||$')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xlim(1, points_to_compute_lr)
        # plt.tight_layout()
        # if save_PDF:
        #     save_fig(args.run_name + 'log_error')
        # else:
        #     plt.show()


if __name__ == '__main__':
    main(args, save_result=save_result, load_run_data_flag=load_run,
         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, plot=plot)
    print('Done!')
    # model = RandBiMod(d, rho, sigma)
