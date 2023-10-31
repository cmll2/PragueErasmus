import utils
import matplotlib.pyplot as plt

OUT_DIR_NO_INFORMED = 'final_partition_no_informed'
OUT_DIR_INFORMED = 'final_informed'
EXP_ID = 'default'

evals, lower, mean, upper = utils.get_plot_data(OUT_DIR_INFORMED, EXP_ID)
plt.figure(figsize=(12, 8))
#utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Informed')
evals, lower, mean, upper = utils.get_plot_data(OUT_DIR_NO_INFORMED, EXP_ID)
utils.plot_experiment(evals, lower, mean, upper, legend_name = 'No informed')
plt.title('Convergence of no informed with log scale')
plt.yscale('log')
plt.legend()
plt.show()