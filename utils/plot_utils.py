
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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


set_default_plot_params()


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
