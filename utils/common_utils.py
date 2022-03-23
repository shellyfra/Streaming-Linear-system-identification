
from datetime import datetime
import os
import numpy as np
import random
import sys
import pickle
import shutil
import glob
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import string


# -----------------------------------------------------------------------------------------------------------#
#  Useful functions
# -----------------------------------------------------------------------------------------------------------#

def save_fig(run_name, base_path='./'):
    ensure_dir(base_path)
    save_path = os.path.join(base_path, run_name)
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    # try:
    #     plt.savefig(save_path + '.pgf', format='pgf', bbox_inches='tight')
    # except:
    #     print('Failed to save .pgf file  \n  tto allow to save pgf files -  $ sudo apt install texlive-xetex')
    print('Figure saved at ', save_path)
# -----------------------------------------------------------------------------------------------------------#


def set_default_plot_params():
    plt_params = {'font.size': 10,
                  'lines.linewidth': 2, 'legend.fontsize': 16, 'legend.handlelength': 2,
                  'pdf.fonttype': 42, 'ps.fonttype': 42,
                  'axes.labelsize': 18, 'axes.titlesize': 18,
                  'xtick.labelsize': 14, 'ytick.labelsize': 14}
    plt.rcParams.update(plt_params)
# -----------------------------------------------------------------------------------------------------------#


def convert_args(args):
    # convert some of the args from string to a python variable

    fields_to_convert = {
        'critic_hiddens', 'actor_hiddens', 'param_grid_def', 'n_traj_grid', 'mdp_def', 'train_sampling_def',
        'learning_rate_def'}

    for key in args.__dict__:
        val = args.__dict__[key]
        if val is not None:
            if key in fields_to_convert:
                try:
                    val = val.replace("\'", "\"")
                    args.__dict__[key] = json.loads(val)
                except:
                    raise Exception("Failed to read " + str(key))
    return args
# -----------------------------------------------------------------------------------------------------------#


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError as e:
        pass  # module doesn't exist
# -----------------------------------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
# -----------------------------------------------------------------------------------------------------------#


def create_result_dir(args):
    # If run_name empty, set according to time
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    run_file_path = sys.argv[0]
    dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    args.result_dir = os.path.join(dir_path, 'results', args.result_dir, args.run_name)
    ensure_dir(args.result_dir)
    message = [
        'Run script: ' + run_file_path,
        'Log file created at ' + time_str,
        'Parameters:', str(args), '-' * 70]
    write_to_log(message, args, mode='w')   # create new log file
    write_to_log('Results dir: ' + args.result_dir, args)
    write_to_log('-' * 50, args)
    # save_git_status(args.result_dir)
    # save_code(args.result_dir)
    save_run_data(args, {'run_time': 0}, verbose=0)
    # pretty_print_args(args)
# ------------------------------------------------------------------------------#


def write_to_log(message, args, update_file=True, mode='a'):
    # mode='a' is append
    # mode = 'w' is write new file
    log_file_path = os.path.join(args.result_dir, 'log') + '.out'
    write_to_file(message, log_file_path, update_file=True, mode='a')
# ------------------------------------------------------------------------------#


def write_to_file(message, log_file_path, update_file=True, mode='a'):
    # mode='a' is append
    # mode = 'w' is write new file
    if not isinstance(message, list):
        message = [message]
    # update log file:
    if update_file:
        with open(log_file_path, mode) as fl:
            for string in message:
                print(string, fl)
    # print to console:
    for string in message:
        print(string)
# -----------------------------------------------------------------------------------------------------------#


def time_now():
    return datetime.now().strftime('%Y\%m\%d, %H:%M:%S')
# -----------------------------------------------------------------------------------------------------------#


def save_best_performance_of_exps(dict, full_path):
    with open(full_path, 'wb') as f:
        pickle.dump(dict, f)


def save_run_data(args, info_dict, verbose=1):
    run_data_file_path = os.path.join(args.result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'wb') as f:
        pickle.dump([args, info_dict], f)
    if verbose == 1:
        write_to_log('Results saved in ' + run_data_file_path, args)
# -----------------------------------------------------------------------------------------------------------#


def load_run_data(result_dir, showParams=True):
    run_data_file_path = os.path.join(result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'rb') as f:
       args, info_dict = pickle.load(f)
    print('Data loaded from ', run_data_file_path)
    if showParams:
        print('Parameters \n', args)
    return args, info_dict
# -----------------------------------------------------------------------------------------------------------#


def load_saved_vars(result_dir):
    run_data_file_path = os.path.join(result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'rb') as f:
        loaded_args, loaded_dict = pickle.load(f)
    print('Loaded run parameters: ' + str(loaded_args))
    print('-' * 70)
    return loaded_args, loaded_dict
# -----------------------------------------------------------------------------------------------------------#


def create_results_backup(result_dir):
    src = os.path.join(result_dir, 'run_data.pkl')
    dst = os.path.join(result_dir, 'backup_run_data.pkl')
    shutil.copyfile(src, dst)
    print('Backup of run data with original grid was saved in ', dst)

# -----------------------------------------------------------------------------------------------------------#


def save_code(save_dir):
    # Create backup of code
    source_dir = os.getcwd()
    dest_dir = save_dir + '/Code_Archive/'
    ensure_dir(dest_dir)

    for filename in glob.glob(os.path.join(source_dir, '*.*')):
        if ".egg-info" not in filename and ".py" in filename:
            shutil.copy(filename, dest_dir)

# -----------------------------------------------------------------------------------------------------------#


def bold(s):
    return '\033[1m ' + s + '\033[0m'
# -----------------------------------------------------------------------------------------------------------#


# def pretty_print_args(args, printFlag=True):
#     from termcolor import colored
#     arg_dict = args.__dict__
#     arg_keys = list(arg_dict.keys())
#     arg_keys.sort(key=lambda k: k.lower())  # sort alphabetically (ignore letter case)
#     s = '{'
#     for key in arg_keys:
#         s += colored(bold('"' + key + '"'), 'blue') + ': ' + str(arg_dict[key]) + ', '
#     s += '}'
#     if printFlag:
#         print(s)
#     return s


