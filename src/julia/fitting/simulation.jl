include("../models/vanilla_hmm.jl")
include("em_hmm.jl")
include("../utils/utils.jl")
using Clustering

n_samples = 2000
obs_dists = [Normal, Normal, Normal]
params = [(0, 1), (5, 1)] # (means, standard deviation) for each state
probs = [0.5, 0.5] # initial transition
A = [0.4 0.6; 0.2 0.8] # transition matrix
y, Z = make_fake_data_hmm(n_samples, obs_dists, params, probs, A) # generate fake data

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[1].imshow(A)
im = ax[2].imshow(pos.Ψ)
ax[1].set_title("True transition matrix")
ax[2].set_title("Estimated transition matrix")

for i in 1:2
    ax[i].set_xticks([])
    ax[i].set_xticklabels([])
    ax[i].set_yticks([])
    ax[i].set_yticklabels([])
end

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

