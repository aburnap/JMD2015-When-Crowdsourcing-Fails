import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./experts_added_results.csv', delimiter=',')
x_experts_added = data[:,0]
num_experts = len(list(set(x_experts_added)))
y_BN_error_vector = data[:,2]
y_BN_average_errors = np.zeros(num_experts)
y_BN_std_errors = np.zeros(num_experts)
for i, num_added in enumerate(set(x_experts_added)):
    ids = np.where(x_experts_added == num_added)
    y_BN_average_errors[i] = np.average(y_BN_error_vector[ids])
    y_BN_std_errors[i] = np.std(y_BN_error_vector[ids])

#----------- PLOTTING --------------------------
plt.errorbar(x=range(num_experts), y=y_BN_average_errors, yerr=y_BN_std_errors, marker='o', markersize=8, color='b', ecolor='black', capsize=4, ls='none', elinewidth=1.5)

plt.title("Mean Evaluation Error vs Number of Added Experts", fontsize=16)
plt.xlabel("Number of Added Experts", fontsize=16)
plt.ylabel("Evaluation Error", fontsize=16)
