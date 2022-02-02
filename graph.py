import numpy as np
import random
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.color_palette()
plt.rcParams["figure.figsize"] = (7,4)
plt.rcParams['font.family'] = "serif"


sel4sel = np.zeros((3, 2000, 6, 20))
linear = np.zeros((3, 2000, 6, 20))
localcomp = np.zeros((3, 2000, 6, 20))
fitness = np.zeros((3, 2000, 6, 20))
novelty = np.zeros((3, 2000, 6, 20))
mincrit = np.zeros((3, 2000, 6, 20))
drift = np.zeros((3, 2000, 6, 20))
mapelite = np.zeros((3, 2000, 6, 20))

for k in range(3):
    for i in range(20):
        if k == 0:
            sel4sel[k,:,:,i] = np.load("res_npy/{}_convex1k.pt_{}.npy".format(k,i))
            linear[k,:,:,i] = np.load("res_npy/{}_convex_linear1k.pt_{}.npy".format(k,i))
        if k == 1:
            sel4sel[k,:,:,i] = np.load("res_npy/{}_hashed1k.pt_{}.npy".format(k,i))
            linear[k,:,:,i] = np.load("res_npy/{}_hashed_linear1k.pt_{}.npy".format(k,i))
        if k == 2:
            sel4sel[k,:,:,i] = np.load("res_npy/{}_deceptive1k.pt_{}.npy".format(k,i))
            linear[k,:,:,i] = np.load("res_npy/{}_deceptive_linear1k.pt_{}.npy".format(k,i))
        fitness[k,:,:,i] = np.load("res_npy/{}_greedy_{}.npy".format(k,i))
        novelty[k,:,:,i] = np.load("res_npy/{}_novelty_{}.npy".format(k,i))
        mincrit[k,:,:,i] = np.load("res_npy/{}_mincrit_{}.npy".format(k,i))
        drift[k,:,:,i] = np.load("res_npy/{}_random_{}.npy".format(k,i))
        localcomp[k,:,:,i] = np.load("res_npy/{}_localcomp_{}.npy".format(k,i))
        mapelite[k,:,0,i] = np.load("res_npy/{}_mapelite_{}.npy".format(k,i))

np.set_printoptions(threshold=sys.maxsize)
        

labels = ["Sel4Sel", "Sel4Sel (Linear)", "Local Competition", "MAP-Elites", "Underlying Fitness", "Novelty", "Minimal Criterion", "Random Drift"]
# h_datas = [x[1] for x in [sel4sel, linear, localcomp, fitness, novelty, mincrit, drift]]
# d_datas = [x[2] for x in [sel4sel, linear, localcomp, fitness, novelty, mincrit, drift]]

# for x in e_datas:
#     print(x[-1,0])


for i,name in enumerate(["Convex", "Hashed", "Deceptive"]):
    t = np.arange(2000)
    datas = [x[i] for x in [sel4sel, linear, localcomp, mapelite, fitness, novelty, mincrit, drift]]
    en = 2
    for label, data in zip(labels, datas):
        en -= 0.1 
        means = np.mean(data[:, 0, :], axis=1)
        mins = np.min(data[:, 0, :], axis=1)
        maxs = np.max(data[:, 0, :], axis=1)
        stds = np.std(data[:, 0, :], axis=1) * 0.5
        # stds = 3
        print("{} & {} $\pm$ {} & {} $\pm$ {} \\\\".format(label, round(np.mean(data[-1, 0, :]), 2), round(np.std(data[-1, 0, :]), 2), round(np.mean(data[-1, 3, :]), 2), round(np.std(data[-1, 3, :]), 2)))
        plt.plot(t, means, label=label,zorder=en)
        plt.fill_between(t, np.maximum(mins, means-stds), np.minimum(maxs, means+stds), alpha=0.3)
    plt.title("{} Bits: Fitness Over Time".format(name))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend(loc='lower right')
    plt.show()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_{}.png".format(name))
    plt.clf()
    
for i,name in enumerate(["Convex", "Hashed", "Deceptive"]):
    t = np.arange(2000)
    datas = [x[i] for x in [sel4sel, linear, localcomp, mapelite, fitness, novelty, mincrit, drift]]
    en = 2
    for label, data in zip(labels, datas):
        en -= 0.1 
        if label == "MAP-Elites":
            plt.plot([],[])
            plt.fill_between([],[],[])
        else:
            means = np.mean(data[:, 3, :], axis=1)
            mins = np.min(data[:, 3, :], axis=1)
            maxs = np.max(data[:, 3, :], axis=1)
            stds = np.std(data[:, 3, :], axis=1) * 2
            plt.plot(t, means, label=label,zorder=en)
            plt.fill_between(t, np.maximum(mins, means-stds), np.minimum(maxs, means+stds), alpha=0.3)
    plt.title("{} Bits: Novelty Over Time".format(name))
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend(loc='lower right')
    plt.show()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("fig_nov_{}.png".format(name))
    plt.clf()