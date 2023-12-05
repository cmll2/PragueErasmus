DIFF = 'continuous_diff'
ADAPTIVE_RANK_LOWER = 'continuous_adaptiveRank'
DEFAULT = 'continuous_default'
DIFF_RANDCR = 'continuous_diffRandCR'
DIFF_RAND = 'continuous_diffRand'
SIM_ANNEAL = 'continuous_SimAnn'
LAMARCK_NOCROSS = 'continuous_LamarckNoCross'
LAMARCK = 'continuous_Lamarck'
DIFF_2500 = 'continuous_diff2500'
DIFF_2500TEST = 'continuous_diff2500test'
SIM_ANNEAL_2500 = 'continuous_SimAnn2500'
LAMARCK_2500 = 'continuous_Lamarck2500'
# method_names = ['Diff', 'Adaptive Rank', 'Default', 'Diff FRand & CR']
# method_dirs = [DIFF, ADAPTIVE_RANK_LOWER, DEFAULT, DIFF_RANDCR]
method_names = ['Diff', 'Diff test']
method_dirs = [DIFF_2500, DIFF_2500TEST]
EXP_ID_1 = 'default.f01'
EXP_ID_2 = 'default.f02'
EXP_ID_3 = 'default.f06'
EXP_ID_4 = 'default.f08'
EXP_ID_5 = 'default.f10'
EXP_IDS = [EXP_ID_1, EXP_ID_2, EXP_ID_3, EXP_ID_4, EXP_ID_5]
fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

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
    plt.savefig('convergence/' + fit_names[i] + '_Diffconvergence.png')
    plt.close()
