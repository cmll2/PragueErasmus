DIFF_NSGA = 'data/multi_diff'
DEFAULT = 'data/multi_default'
DIFF_SUM = 'data/multi_diff_sum'
DIFF_SUM1 = 'data/multi_diff_sum1'
DIFF_HYPER = 'data/multi_diff_criterion'
HYPER = 'data/multi_criterion'
method_names = ['default', 'diff_nsga', 'diff_sum', 'diff_hypervolume', 'default_cd_hypervolume']
method_dirs = [DEFAULT, DIFF_NSGA, DIFF_SUM1, DIFF_HYPER, HYPER]
#zdt1,2,3,4,6
EXP_ID_1 = 'default.ZDT1'
EXP_ID_2 = 'default.ZDT2'
EXP_ID_3 = 'default.ZDT3'
EXP_ID_4 = 'default.ZDT4'
EXP_ID_5 = 'default.ZDT6'
EXP_IDS = [EXP_ID_1, EXP_ID_2, EXP_ID_3, EXP_ID_4, EXP_ID_5]
fit_names = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']

def chose_linestyle(i):
    if i == 0:
        return '-'
    elif i == 1:
        return '--'
    elif i == 2:
        return ':'
    elif i == 3:
        return '-.'
    else:
        return '-'
import utils

import matplotlib.pyplot as plt

for i in range(len(EXP_IDS)):
    plt.figure(figsize=(12, 8))
    for j in range(len(method_dirs)):
        linestyle = chose_linestyle(j)
        evals, lower, mean, upper = utils.get_plot_data(method_dirs[j], EXP_IDS[i])
        utils.plot_experiment(evals, lower, mean, upper, legend_name = method_names[j], linestyle = '-')
    plt.title('Convergence of different methods for ' + fit_names[i])
    plt.yscale('log')
    plt.legend()
    #save
    plt.savefig('convergence/' + fit_names[i] + '_convergence.png')
    plt.close()
