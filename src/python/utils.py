import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from scipy.special import expit

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

def simulate_data_hmm_bern(n_samples, likelihood_dists, params, pi, A):
    # initialize matrices
    Z = np.zeros(n_samples, dtype=int)
    y = np.zeros(n_samples)

    # get first time step
    Z[0] = _get_Z(pi)
    y[0] = likelihood_dists[Z[0]](params[Z[0]]).rvs()

    for t in range(1, n_samples):
        Z[t] = _get_Z(A[Z[t-1], :])
        y[t] = likelihood_dists[Z[t]](params[Z[t]]).rvs()

    return y, Z

def simulate_data_hmm_glm_gauss(n_samples, likelihood_dists, W, X, pi, A, stdev):
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



def simulate_data_hmm_glm_bern(n_samples, likelihood_dists, W, X, pi, A):
    Z = np.zeros(n_samples, dtype=int)
    y = np.zeros(n_samples)

    Z[0] = _get_Z(pi)
    theta_0 = expit(X[0, :] @ W[:, Z[0]])
    y[0] = likelihood_dists[Z[0]](theta_0).rvs()

    for t in range(1, n_samples):
        Z[t] = _get_Z(A[Z[t-1], :])
        theta_t = expit(X[t, :] @ W[:, Z[t]])
        y[t] = likelihood_dists[Z[t]](theta_t).rvs()

    return y, Z

def _indicator_y(y, k):
    return (y == k).astype(int)


def _indicator_Y(y, k_max):
    return np.vstack([_indicator_y(y, k) for k in range(k_max)])


# def comps(theta, X, y):
#     N = np.shape(y)[0]
#     K = len(np.unique(y)) - 1
#     P = np.shape(X)[1]
#     W = np.reshape(theta, (P, K))
#     xproj = X @ W
#     row_max = np.max(xproj, axis=1, keepdims=True)
#     xproj = xproj - row_max
#     exp_xproj = np.exp(xproj)
#     log_norm = np.log(np.sum(exp_xproj, axis=1, keepdims=True))
#     Y = _indicator_Y(y, K).T
#     return Y, log_norm, exp_xproj, xproj, P, K


# def cost(theta, X, y):
#     Y, log_norm, _, xproj, _, _ = comps(theta, X, y)
#     cost = Y * xproj - Y * log_norm
#     return -cost.sum(1).sum()


#from scipy.linalg import block_diag

def cost_jax_softmax(theta, X, y):
    P = jnp.shape(X)[1]
    N = np.shape(y)[0]
    K = len(np.unique(y))
    xproj = X @ theta.reshape((P, K))
    exp_xproj = jnp.exp(xproj)
    log_norm = jnp.log(jnp.sum(exp_xproj, axis=1, keepdims=True))
    Y = _indicator_Y(y, K).T
    cost = Y * xproj - Y * log_norm
    return -cost.sum(1).sum()
   
"""
optimizing via scipy, "SLSQP" works well

"""
def optimize_softmax(X, y, method):
    N, P = X.shape
    K = len(np.unique(y)) # number of classes
    x0 = jnp.zeros(P*K)
    min = minimize(cost_jax_softmax, x0, args=(X, y), jac=grad(cost_jax_softmax), method=method)
    return min

def cost_jax_logistic(theta, gamma_k, X, y):
    xproj = X @ theta[:, None]
    p_z = gamma_k / jnp.sum(gamma_k)
    L = -(jnp.multiply(p_z, y[:, jnp.newaxis])).T @ xproj
    return (L.ravel() + jnp.multiply(p_z, jnp.log(1 + jnp.exp(xproj))).sum())[0]


def optimize_logistic(X, y, gamma_k, method):
    _, P = X.shape
    x0 = jnp.zeros(P)
    opt = minimize(cost_jax_logistic, x0, args=(gamma_k, X, y), 
                      jac=grad(cost_jax_logistic), 
                      hess=hessian(cost_jax_logistic), method=method)
    return opt


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
likelihood_dists = [ss.bernoulli, ss.bernoulli]
pi = [0.5, 0.5]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])
W = np.array([[-2.3, 1], 
              [-4.2, 0.5], 
              [-3.5, 1.5]])

X = np.random.random((n_samples, 3))
y, Z = simulate_data_hmm_glm_gauss(n_samples, likelihood_dists, W, X, pi, A, [1.2, 0.3])

y, Z = simulate_data_hmm_glm_bern(n_samples, likelihood_dists, W, X, pi, A)

est, f, b, d = fit_hmm_glm_em(y, likelihood_dists, X)


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, 
                           n_redundant=0, n_classes=2, random_state=1)
# summarize the dataset
N, P = X.shape
K = 1
theta_mat = np.random.random((P, K))

test = optimize_softmax(X, y, "SLSQP")
 

y, Z = simulate_data_hmm_bern(n_samples, likelihood_dists, [0.3, 0.8], pi, A)
est, f, b, d = fit_hmm_em(y, likelihood_dists)
