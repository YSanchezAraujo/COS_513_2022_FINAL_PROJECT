import numpy as numpy
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib
matplotlib.rcParams.update({'font.size': 16})



data_path = "/Users/yoelsanchezaraujo/Desktop"
mouse = 29
day = 1

X = np.load("fip_29_X.npy")
Y = np.load("fip_29_Y_feed.npy")
cstim = np.load("cstim.npy")
cfeed = np.load("cfeed.npy")
b_stim = np.load("bools_stim.npy")
b_feed = np.load("bools_feed.npy")
lik_dist = [ss.norm, ss.norm]
y_idx = 0
idx_use = ~np.isnan(Y[:, y_idx])
est2, f, b, d, ll = fit_hmm_glm_em(Y[idx_use, y_idx], lik_dist, X[idx_use, :], max_iter=50)

LLs = np.zeros((49, 2))
LLs[:, 0] = ll[1:]

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
    plt.plot(est2["W"].T[:, c], "-o", lw=5, color=col[c], linestyle=linestyles[c])
plt.xticks([0, 1], ["state 1", "state 2"], fontsize=20)
plt.legend(labs, fontsize=20)
plt.title("Mouse 1 NACC estimated input weights", fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("weight values", fontsize=20)
plt.savefig("W_states__feedback_2fip29.png", dpi=200, bbox_inches="tight")

# 
# GAMMA PLOT
#
pz = est2["gamma"]
pz = pz / np.sum(pz, axis=1, keepdims=True)
a = 0
b = cstim[0, 0]
plt.plot(pz[a:b, :], lw=3)
for k in cstim[:, 0]:
    plt.axvline(k, linestyle="--", color="black", lw=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("sessions", fontsize=20)
plt.ylabel("P(state = k)", fontsize=20)
plt.legend(["state 1", "state 2", "state 3"], fontsize=20)
plt.title("Mouse 1 NACC estimated state probabilities", fontsize=20)
plt.savefig("gammaall_S4_ses.png", dpi=200, bbox_inches="tight")


fig, ax = plt.subplots(nrows=2, ncols=2)
a0, b0 = cfeed[0, 0], cfeed[1, 0]
a1, b1 = cfeed[1, 0], cfeed[2, 0]
a2, b2 = cfeed[2, 0], cfeed[3, 0]
a3, b3 = cfeed[3, 0], cfeed[4, 0]
ax[0, 0].plot(pz[a0:b0, :], lw=3)
ax[0, 1].plot(pz[a1:b1, :], lw=3)
ax[1, 0].plot(pz[a2:b2, :], lw=3)
ax[1, 1].plot(pz[a3:b3, :], lw=3)
ax[0, 0].legend(["state 1", "state 2"])
ax[0, 0].set_title("session 1")
ax[0, 1].set_title("session 2")
ax[1, 0].set_title("session 3")
ax[1, 1].set_title("session 4")
fig.text(0.5, 0.04, 'trials within sessions', ha='center')
fig.text(0.04, 0.5, 'P(state = k)', va='center', rotation='vertical')
plt.savefig("first_1to4_days_2state__feedback_gammas.png", dpi=200, bbox_inches="tight")

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
plt.savefig("Y_by_state_2states.png", dpi=200, bbox_inches="tight")

plt.imshow(est2["Phi"])
plt.colorbar()
plt.title("Estimated transition matrix")
plt.xticks([0, 1], ["state 1", "state 2"])
plt.yticks([0, 1], ["state 1", "state 2"])
plt.savefig("trans_mat_2state_feedback.png", dpi=200, bbox_inches="tight")


z_idx = np.argmax(pz, axis=1)
Y[idx_use, y_idx][z_idx == 0]
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(Y[idx_use, y_idx][z_idx == 0], lw=3)
ax[1].plot(Y[idx_use, y_idx][z_idx == 1], lw=3)
ax[2].plot(Y[idx_use, y_idx][z_idx == 2], lw=3)


plt.hist(Y[idx_use, y_idx][z_idx == 0], edgecolor="white")
plt.hist(Y[idx_use, y_idx][z_idx == 1], edgecolor="white")
#plt.hist(Y[idx_use, y_idx][z_idx == 2], edgecolor="white", alpha=0.5)

plt.legend(["state 1", "state 2"])
plt.ylabel("counts")
plt.xlabel("Brain signal")
plt.title("Brain signal by most likely state")
plt.savefig("2state_hist_feedback.png", dpi=200, bbox_inches="tight")

