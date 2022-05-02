import numpy as numpy
import os
import matplotlib.pyplot as plt
import scipy.stats as ss

data_path = "/jukebox/witten/yoel/data"
mouse = 29
day = 1

def load_data(path, mouse, day):
    X_load_str = "X_fip_" + str(mouse) + "_day_" + str(day) + ".npy"
    y_load_str = "Y_fip_" + str(mouse) + "_day_" + str(day) + ".npy"
    X = np.load(os.path.join(path, X_load_str))
    Y = np.load(os.path.join(path, y_load_str))
    return {"X":X, "Y":Y}

dat = load_data(data_path, mouse, day)    

X = np.load("fip_29_X.npy")
Y = np.load("fip_29_Y.npy")
use_samps = np.load("use_samps_fip29.npy")
num_samps = np.load("num_samps_fip29.npy")
lik_dist = [ss.norm, ss.norm, ss.norm]
y_idx = 0
idx_use = ~np.isnan(Y[:, y_idx])
est, f, b, d = fit_hmm_glm_em(Y[idx_use, y_idx], lik_dist, X[idx_use, :], max_iter=20)

cons = ["6.25%", "12.5%", "25%", "100%"]
rcons = ["right " + i for i in cons]
lcons = ["left " + i for i in cons]
col_ipsi = ["#94d2bd", "#0a9396","#005f73", "#001219"]    
col_contra = ["#f48c06", "#dc2f02", "#9d0208", "#370617"]
col = [j for i in [["green", "purple"] , col_ipsi, col_contra] for j in i]

##
# W PLOT BY STATE
##
labs = [j for i in [ ["choice", "rewarded"], rcons, lcons] for j in i]
linestyles = ["--", "--", "-", "-", "-", "-","-", "-", "-", "-"]
for c in range(10):
    plt.plot(est["W"].T[:, c], "-o", lw=5, color=col[c], linestyle=linestyles[c])
plt.xticks([0, 1, 2], ["state 1", "state 2", "state 3", fontsize=20)
plt.legend(labs, fontsize=20)
plt.title("Mouse 1 NACC estimated input weights", fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("weight values", fontsize=20)
plt.savefig("W_states_4fip29.png", dpi=200, bbox_inches="tight")

# 
# GAMMA PLOT
#
xi = np.cumsum(num_samps[use_samps[:, y_idx], y_idx])
plt.plot(est["gamma"], lw=3)
for k in xi:
    plt.axvline(k, linestyle="--", color="black", lw=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("sessions", fontsize=20)
plt.ylabel("P(state = k)", fontsize=20)
plt.legend(["state 1", "state 2", "state 3"], fontsize=20)
plt.title("Mouse 1 NACC estimated state probabilities", fontsize=20)
plt.savefig("gammaall_S4_ses.png", dpi=200, bbox_inches="tight")

matplotlib.rcParams.update({'font.size': 22})

#
# Psychometric curves by day
#

pcurves = np.load("fip_29_all_days_pcurve_1.npy")
fig, ax = plt.subplots(nrows=2, ncols=1)
nses = 19
x_t = list(range(1, nses+1))
ax[0].plot(x_t, pcurves[:, 4, 1, 0], "-o", lw=3)
ax[0].plot(x_t, pcurves[:, 5, 1, 0], "-o", lw=3)
ax[0].plot(x_t, pcurves[:, 6, 1, 0], "-o", lw=3)
ax[0].plot(x_t, pcurves[:, 7, 1, 0], "-o", lw=3)

# choice pos1, index: [days, contrastlevel, prob correct, side]
ax[1].plot(x_t, pcurves[:, 3, 1, 1], "-o", lw=3)
ax[1].plot(x_t, pcurves[:, 2, 1, 1], "-o", lw=3)
ax[1].plot(x_t, pcurves[:, 1, 1, 1], "-o", lw=3)
ax[1].plot(x_t, pcurves[:, 0, 1, 1], "-o", lw=3)

ax[0].legend(["6.25%", "12.5%", "25%", "100%"], fontsize=20)
ax[1].set_xlabel("sessions", fontsize=20)
ax[0].set_title("Correct turns to the left", fontsize=20)
ax[1].set_title("Correct turns to the right", fontsize=20)
plt.savefig("prop_turns_psych.png", dpi=200, bbox_inches="tight")


fig, ax = plt.subplots(nrows=3, ncols=1)

for i in range(3):
    ax[i].plot(Y[idx_use, 0][z == i][:200])
    if i != 3:
         ax[i].set_xticks([])
         ax[i].set_xticklabels([])

ax[0].set_ylabel("state 1")
ax[1].set_ylabel("state 2")
ax[2].set_ylabel("state 3")
ax[2].set_xlabel("samples")
ax[0].set_title("NACC signal by most likely state")
plt.savefig("Y_by_state_3states.png", dpi=200, bbox_inches="tight")

plt.imshow(est["Phi"])
plt.colorbar()
plt.title("Estimated transition matrix")
plt.savefig("trans_mat_3state.png", dpi=200, bbox_inches="tight")
