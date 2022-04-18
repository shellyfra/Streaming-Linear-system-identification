
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utils.common_utils import save_fig, load_run_data
# plt.style.use('ggplot')


def set_default_plot_params():
    # mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['text.usetex'] = True
    plt.rcParams["savefig.format"] = 'pdf'
    # plt.rcParams['axes.edgecolor'] = 'grey'
    # plt.rcParams['axes.edgecolor'] = 'grey'
    # plt.rcParams['axes.linewidth'] = 2


# set_default_plot_params()

# np.arange(0, avg_arr.shape[0] + 1 + subsample)[::subsample]
def plot_convergence(avg_arr, std_arr=None, xaxis=None, n_reps=1, subsample=100,
                     mode='ci', label=None, semilogx=False):
    if xaxis is None:
        xaxis = np.arange(1, avg_arr.shape[0]+1)[::subsample]
    if semilogx:
        p = plt.semilogx(xaxis, avg_arr[::subsample], linewidth=2, label=label)
    else:
        p = plt.plot(xaxis, avg_arr[::subsample], linewidth=2, label=label)
    if mode == 'ci':
        factor = 1.96 / np.sqrt(n_reps)
    elif mode == 'std':
        factor = 1.
    else:
        raise AssertionError('Unrecognized plotting mode')
    if n_reps > 1 and std_arr is not None:
        lb, ub = avg_arr[::subsample] - factor * std_arr[::subsample], \
                 avg_arr[::subsample] + factor * std_arr[::subsample]
        plt.gca().fill_between(xaxis, lb, ub, facecolor=p[0].get_color(), alpha=0.3)


# def plot_error_bars(avg_arr, std_arr=None, xaxis=None, n_reps=1,
#                     mode='ci', w=0.2):
#     if xaxis is None:
#         xaxis = np.arange(1, avg_arr.shape[0]+1)
#
#     if mode == 'ci':
#         factor = 1.96 / np.sqrt(n_reps)
#     elif mode == 'std':
#         factor = 1.
#     else:
#         raise AssertionError('Unrecognized plotting mode')
#
#     plt.bar(xaxis, avg_arr, width=w, yerr=factor * std_arr)


def plot_error_bars(avg_arr, std_arr, x_ticks, n_reps=1, label=None, width=None, color=None):
    factor = 1.96 / np.sqrt(n_reps)     # 95% confidence interval
    if width is None:
        plt.bar(x_ticks, avg_arr, yerr=factor * std_arr, label=label, color=color)
    else:
        plt.bar(x_ticks, avg_arr, yerr=factor * std_arr, width=width, label=label, color=color)


def plot_all_dict_instances():
    folders = ["SGD", "SGD_ER", "SGD_RER", "MAG"]
    for name in folders:
        res_dir = 'results\\RandBiMod\\' + str(name)
        try:
            args2, info_dict2 = load_run_data(res_dir)
        except FileNotFoundError:
            continue
        error_avg_mem, error_std_mem = info_dict2["error_avg"], info_dict2["error_std"]
        if name == "SGD_RER":
            plot_convergence(error_avg_mem, error_std_mem, n_reps=args2.n_reps, subsample=1, label=name,
                             xaxis=np.arange(args2.lr_params["B"], args2.T, args2.lr_params["B"] + args2.u))
        elif name == 'SGD':
            OLS_error_avg, OLS_error_std = info_dict2["OLS_error_avg"], info_dict2["OLS_error_std"]
            plot_convergence(error_avg_mem, error_std_mem, n_reps=args2.n_reps, subsample=args2.subsample, label=args2.run_name)
            plot_convergence(OLS_error_avg, OLS_error_std, xaxis=np.arange(1, error_avg_mem.shape[0] + 1, args2.subsample),
                             subsample=1, n_reps=args2.n_reps, label='OLS')
        else:
            plot_convergence(error_avg_mem, error_std_mem, n_reps=args2.n_reps, subsample=args2.subsample, label=name)
    plt.xlim(1, args2.T)
    plt.xlabel('Num. iterations')
    plt.ylabel(r'$||\overline{A}_{t} - A^*||$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_instance(args, info_dict, plot, save_PDF):
    error_avg, error_std = info_dict["error_avg"], info_dict["error_std"]
    if args.optimizer == 'SGD':
        OLS_error_avg, OLS_error_std = info_dict["OLS_error_avg"], info_dict["OLS_error_std"]
    # ----- Plot figures ----- #
    if plot or save_PDF:
        if args.optimizer == 'SGD_RER':
            x_axis = np.arange(args.lr_params["B"], args.T, args.lr_params["B"] + args.u)
            # x_axis = np.arange(1, args.T + 1)
            plot_convergence(error_avg, error_std, n_reps=args.n_reps, subsample=1, label=args.run_name,
                             xaxis=x_axis)
        elif args.optimizer == 'SGD':
            plot_convergence(error_avg, error_std, n_reps=args.n_reps, subsample=args.subsample, label=args.run_name)
            plot_convergence(OLS_error_avg, OLS_error_std, xaxis=np.arange(1, error_avg.shape[0] + 1, args.subsample), subsample=1, n_reps=args.n_reps, label="OLS")
        else:
            plot_convergence(error_avg, error_std, n_reps=args.n_reps, subsample=args.subsample, label=args.run_name)
        # x = np.arange(1, disp_errors.shape[0]+1)
        # plt.scatter(x, disp_errors.reshape(-1))
        plt.xlim(1e2, args.T)
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
