import pickle
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style("white")

num_trials = 5

methods = ['_residual.pkl', '_onept.pkl']

meanl, maxl, minl = [], [], []
for method in methods:
    # load data
    ds_directory = './rewards/'
    exp_name = 'zomaddpg_'

    ls = []
    for ind_trial in range(num_trials):
        filename = ds_directory + exp_name + str(ind_trial) + method
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            ls.append(pickle.load(f))

    y = np.array(ls)

    mean, maxd, mind = np.mean(y, axis=0), np.max(y, axis=0), np.min(y, axis=0)

    meanl.append(mean)
    maxl.append(maxd)
    minl.append(mind)

# print('/n data type is {}'.format(type(mean)))
# print('/n data is {}'.format(mean))

x = np.arange(0, len(mean))

fs = 14
fig, ax = plt.subplots(1, 1, sharex=True)
st = sns.axes_style("whitegrid")
# ax.set_yscale('log')
plt.ylim([-500, -300])
plt.xlim([0, 50])
plt.xticks(np.arange(0, stop=51, step=10), np.arange(
    0, stop=50001, step=10000), fontsize=fs)
plt.yticks(fontsize=fs)

for mean, maxd, mind in zip(meanl, maxl, minl):
    # ax.plot(x, mean, lw=3)
    # ax.fill_between(x, mind, maxd, alpha=0.2, interpolate=True)
    ax.plot(x[-(len(x)-1):], mean[-(len(x)-1):], lw=3)
    ax.fill_between(x[-(len(x)-1):], mind[-(len(x)-1):],
                    maxd[-(len(x)-1):], alpha=0.2, interpolate=True)

plt.legend(['One-point Residual Feedback', 'One-point Feedback'],
           fontsize=fs, loc='lower right')
plt.xlabel('$Query\ Complexity$', fontsize=16)
plt.ylabel(r'$J(\theta_t)$', fontsize=16)

plt.tight_layout()
plt.savefig(ds_directory+'curve.pdf')
