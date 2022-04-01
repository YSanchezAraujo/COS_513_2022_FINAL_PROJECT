import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans

def _get_Z(p):
    dist = ss.multinomial
    idx = np.where(dist(n=1, p=p).rvs().ravel())
    return idx[0].astype(np.int64)

def _uniform_prob(n):
    return np.array([1/n for i in range(n)])

def _normalize_z(x):
    Z = np.sum(x)
    return x / Z, Z

def init_kmeans(y, n_clusts):
    km = KMeans(n_clusters=n_clusts).fit(y[:, np.newaxis])
    mu = km.cluster_centers_
    sd = np.array([np.std(y[km.labels_ == i]) for i in np.unique(km.labels_)])
    params = [(mu[k], sd[k]) for k in range(n_clusts)]
    return params


def simulate_data_hmm(n_samples, likelihood_dists, params, pi, A):
    # initialize matrices
    Z = np.zeros(n_samples, dtype=int)
    y = np.zeros(n_samples)

    # get first time step
    Z[0] = _get_Z(pi)
    y[0] = likelihood_dists[Z[0]](*params[Z[0]]).rvs()

    for t in range(1, n_samples):
        Z[t] = _get_Z(A[Z[t-1], :])
        y[t] = likelihood_dists[Z[t]](*params[Z[t]]).rvs()

    return y, Z

# testing
n_samples = 100
likelihood_dists = [ss.norm, ss.norm]
params = [(0, 1), (10, 1)]
pi = [0.3, 0.7]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])

y, Z = simulate_data_hmm(n_samples, likelihood_dists, params, pi, A)

test = fit_hmm_em(y, likelihood_dists)
