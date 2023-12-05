OUT_DIR_1 = 'continuous_diffRandCR'
OUT_DIR_2 = 'continuous_SimAnn'
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
plt.figure(figsize=(12, 8))
for i in range(len(EXP_IDS)):
    linestyle = chose_linestyle(i)
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR_1, EXP_IDS[i])
    utils.plot_experiment(evals, lower, mean, upper, legend_name = fit_names[i] + 'diff' , linestyle = linestyle)
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR_2, EXP_IDS[i]) 
    utils.plot_experiment(evals, lower, mean, upper, legend_name = fit_names[i] + ' sim ann', linestyle = linestyle)
plt.title('Convergence of different functions')
plt.yscale('log')
plt.legend()
plt.show()
