import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
likelihood_dists = [ss.norm, ss.norm]
pi = [0.5, 0.5]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])
W = np.array([[-2.3, 1], 
              [-4.2, 0.5], 
              [-3.5, 1.5]])

X = np.random.random((n_samples, 3))
y, Z = simulate_data_hmm_glm_gauss(n_samples, likelihood_dists, W, X, pi, A, [1.2, 0.3])
est, f, b, d = fit_hmm_glm_em(y, likelihood_dists, X, max_iter=500)

plt.imshow(est["W"])
plt.colorbar()
plt.title("estimated input weight matrix")
plt.xticks([0, 1], ["state 1", "state 2"])
plt.ylabel("weight values")

plt.bar([0, 1], est["stdev"])
plt.bar([0, 1], [0.3, 1.2])

plt.imshow(est["Phi"])
plt.colorbar()
plt.title("estimated transition matrix")
plt.xticks([0, 1], ["state 1", "state 2"])
