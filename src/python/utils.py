import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn import linear_model


def lm(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model.coef_

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


def simulate_data_hmm_glm(n_samples, likelihood_dists, W, X, pi, A, stdev):
    Z = np.zeros(n_samples, dtype=int)
    y = np.zeros(n_samples)

    Z[0] = _get_Z(pi)
    mu_0, sig_0 = X[0, :] @ W[:, Z[0]], stdev[Z[0]]
    y[0] = likelihood_dists[Z[0]](mu_0, sig_0).rvs()

    for t in range(1, n_samples):
        Z[t] = _get_Z(A[Z[t-1], :])
        mu_t = X[t, :] @ W[:, Z[t]]
        sig_t = stdev[Z[t]]
        y[t] = likelihood_dists[Z[t]](mu_t, sig_t).rvs()

    return y, Z




# testing
n_samples = 1000
likelihood_dists = [ss.norm, ss.norm]
params = [(0, 1), (10, 1)]
pi = [0.3, 0.7]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])

y, Z = simulate_data_hmm(n_samples, likelihood_dists, params, pi, A)

est, f, b, d = fit_hmm_em(y, likelihood_dists)


# testing hmm_glm simulator

n_samples = 1000
likelihood_dists = [ss.norm, ss.norm]
params = [1, 0.5]
pi = [0.5, 0.5]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])
W = np.array([[10, 1], 
              [4, 0.5], 
              [5.5, 1.5]])
              
X = np.random.random((n_samples, 3))
y, Z = simulate_data_hmm_glm(n_samples, likelihood_dists, W, X, pi, A, [1, 1])


est, f, b, d = fit_hmm_glm_em(y, likelihood_dists, X)
