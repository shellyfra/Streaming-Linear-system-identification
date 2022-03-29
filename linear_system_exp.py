import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit
import time
from copy import deepcopy
from algorithms_v2 import SGD
from envs.simple_chain import SimpleChainMNISTLogisticRegression, SimpleChainLinearRegression
from envs.linear_system import *
from utils.common_utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, \
    set_default_plot_params, save_fig
from utils.plot_utils import plot_convergence
from utils.projection_utils import project_l2_ball
from utils.learning_utils import create_averages_array
from scipy import interpolate
from scipy.linalg import eigh

set_default_plot_params()

# Option to load/save
load_run = False  # False/True If true just load results from dir, o.w run simulations
result_dir_to_load = ''
save_result = False

# Plot options
save_PDF = False  # False/True - save figures as PDF file in the result folder
plot = True

# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------
args = argparse.Namespace()

# ----- Run Parameters ---------------------------------------------#
args.exp = 'LinearSystemSGD'  # '____' | '_____' | '____'
if args.exp == 'LinearSystemSGD':
    args.d = 5  # A* matrix size
    args.rho = 0.9  # values in the the eigenvalue are rho and rho/3
    args.sigma = 0.001  # noise distribution
    args.T = 1000  # horizon
    args.X0 = None  # TODO: check if we want to initialize it
    # args.gap_size = 10
    # args.buffer_size = 10*args.gap_size

# if args.exp == 'LinearRegression':
#     args.result_dir = 'linear_regression_final_long'
#     args.n = 500  # Num of data points
#     args.dim = 100  # Data dimensionality
#     args.radius = 5
#     # args.sigma = 1 / np.sqrt(args.dim)
#     # args.sigma = 0.5
#     args.sigma = 1
# elif args.exp == 'MNIST':
#     args.result_dir = 'mnist'
else:
    raise NotImplementedError
args.run_name = 'LinearSystems'  # 'Name of dir to save results in (if empty, name by time)'
args.seed = 41  # Random seed
args.n_reps = 1  # Number of experiment repetitions. Default: LinearRegression - 10, MNIST - 20
args.evaluate_on_average = False

args.optimizer = 'SGD'  # 'SGD' | 'SGD_MLMC' | 'SGD_DD'
args.encode_mix_time_in_lr = False  # if True -- use 1/sqrt(tau) as scaling. ONLY FOR SGD!
args.lr_params = {'type': 'SGDLinearSystems'}
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
    if args.exp == 'LinearSystemSGD':
        X0 = 2*np.random.randn(args.d).reshape(-1, 1)
        objective = RandBiMod(d=args.d, rho=args.rho, sigma=args.sigma, X0=X0)
    else:
        raise NotImplementedError

    if args.optimizer == 'SGD':
        optimizer = SGD(w_init=objective.A, lr_params=args.lr_params, T=args.T, momentum_def=None,
                        beta=None, grad_clip=None)
    else:
        raise NotImplementedError
    optimizer.run(T=args.T, objective=objective, project=project)
    optimal_A = objective.A_star
    points_to_compute_lr = int(np.floor(2*np.log(args.T)))
    points = np.hstack(objective.points)
    A_matrix_array = np.array(optimizer.iterates)
    # iterates = np.hstack(optimizer.iterates)
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
    #plt.plot(observed_samples_axis, points, 'o', color='black')
    return optimal_A, points, A_matrix_array

def compute_param_error(optimal_A, X_points, A_matrix_array, T):
    # add X_points for future usage
    # compute A moving avg
    num_of_iterations = X_points.shape[1]  # without first 2logT for the lr
    errors = []
    errors_logT = []
    logT = int(np.round(np.log2(T)))
    for i in range(num_of_iterations):
        test = np.stack(A_matrix_array[:i + 1])
        A_moving_avg = test.mean(axis=0)
        err_val = A_moving_avg - optimal_A
        lambda_max = eigh(err_val, eigvals_only=True)[-1]
        A_norm = np.linalg.norm(err_val, 'fro')
        errors.append(A_norm)
        #todo: check if log10 or log 2
        if i >=logT:
            A_moving_avg_logT = A_matrix_array[logT:i + 1].mean(axis=0)  #todo: check ! "maintain a running tail average at the end of each of the subsequent buffers"
            err_val_logT = A_moving_avg_logT - optimal_A
            lambda_max_logT = eigh(err_val_logT, eigvals_only=True)
            # A_norm_logT = np.sqrt(lambda_max_logT * (np.matmul(err_val_logT, err_val_logT.T)))
            A_norm_logT = np.linalg.norm(err_val, 'fro')
            errors_logT.append(A_norm_logT)
    return errors, errors_logT

def run_simulations(arguments, save_result):
    if save_result:
        create_result_dir(arguments)
        write_to_log('Creating results directory', arguments)

    A_matrixes_all_runs, optimal_A_all_runs, X_points_all_runs = [], [], []
    optimal_A, X_points, A_matrix_array = None, None, None
    start_time = time.time()
    # ----- Run simulations ------ #
    for i_rep in range(arguments.n_reps):
        set_random_seed(arguments.seed + i_rep)
        optimal_A, X_points, A_matrix_array = single_simulation(arguments)
        A_matrixes = np.stack(A_matrix_array, axis=0)  # todo: need to check!
        optimal_A_all_runs = np.stack(optimal_A, axis=0) # todo: need to check!
        X_points_all_runs = np.stack(X_points, axis=0)
        print('Rep {} is done! Elapsed time: {:.3f}[s]'.format(i_rep + 1, time.time() - start_time))
        # todo: how to compute the average for multiple repetitions ?

    disp_errors, disp_errors_logT = compute_param_error(optimal_A, X_points, A_matrix_array, arguments.T)
    return np.array(disp_errors).reshape(-1, 1), np.array(disp_errors_logT).reshape(-1, 1)
    #     loss_mat.append(loss)
    #     suboptimality_mat.append(suboptimality)
    #
    # loss_mat = np.vstack(loss_mat)
    # suboptimality_mat = np.vstack(suboptimality_mat)
    #
    # info_dict = {'loss_avg': loss_mat.mean(axis=0),
    #              'loss_std': loss_mat.std(axis=0),
    #              'suboptimality_avg': suboptimality_mat.mean(axis=0),
    #              'suboptimality_std': suboptimality_mat.std(axis=0)
    #              }

def main(args, save_result=True, load_run_data_flag=False, result_dir_to_load='',
         save_PDF=False, plot=True):
    if load_run_data_flag:
        args, disp_errors, disp_errors_logT = load_run_data(result_dir_to_load)
    else:
        disp_errors, disp_errors_logT = run_simulations(args, save_result)

    points_to_compute_lr = int(np.floor(2*np.log(args.T)))

    # ----- Plot figures ----- #
    if plot or save_PDF:
        plot_convergence(disp_errors, n_reps=args.T, subsample=10)
        # x = np.arange(1, disp_errors.shape[0]+1)
        # plt.scatter(x, disp_errors.reshape(-1))
        plt.xlabel('Num. iterations')
        plt.ylabel(r'$||\bar{A}_{0t} - A^*||$')
        plt.yscale('log')
        plt.tight_layout()
        if save_PDF:
            save_fig(args.run_name + '_error')
        else:
            plt.show()
        plot_convergence(disp_errors_logT, n_reps=args.T - int(np.floor(np.log2(args.T))), subsample=10) # semilogx=True ??
        plt.xlabel('Num. iterations')
        plt.ylabel(r'$||\bar{A}_{log(T)t} - A^*||$')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1, points_to_compute_lr)
        plt.tight_layout()
        if save_PDF:
            save_fig(args.run_name + 'log_error')
        else:
            plt.show()


if __name__ == '__main__':
    main(args, save_result=save_result, load_run_data_flag=load_run,
         result_dir_to_load=result_dir_to_load, save_PDF=save_PDF, plot=plot)
    print('Done!')
    # model = RandBiMod(d, rho, sigma)
