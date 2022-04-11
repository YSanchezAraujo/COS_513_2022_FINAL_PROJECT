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


def _indicator_y(y, k):
    return (y == k).astype(int)


def _indicator_Y(y, k_max):
    return np.vstack([_indicator_y(y, k) for k in range(k_max)])


def get_softmax_comps(X, theta_mat, y):
    _, P = X.shape
    P2, K = theta_mat.shape
    assert P == P2
    x_dot_theta = X @ theta_mat
    x_dot_theta = x_dot_theta - np.max(x_dot_theta, axis=1)[:, np.newaxis]
    # 709 is float64 precision
    #exp_x_dot_theta = np.exp(np.clip(x_dot_theta, -709.78, 709.78))
    exp_x_dot_theta = np.exp(x_dot_theta)
    exp_x_dot_theta_norm = np.sum(exp_x_dot_theta, axis=1)
    prob = exp_x_dot_theta / exp_x_dot_theta_norm[:, np.newaxis]
    # log normalizer
    log_norm = np.log(exp_x_dot_theta_norm)
    log_prob = x_dot_theta - log_norm[:, np.newaxis]
    delta_yk = _indicator_Y(y, K)
    cost = -np.sum(np.multiply(delta_yk, log_prob.T))
    return cost, delta_yk, log_prob, prob


def softmax_cost(theta, X, y):
    N, P = X.shape
    K = len(np.unique(y))
    cost, _, _, _ = get_softmax_comps(X, theta.reshape(P, K), y)
    return cost


def grad_softmax(theta, X, y):
    N, P = X.shape
    K = len(np.unique(y))
    _, delta_yk, _, prob = get_softmax_comps(X, theta.reshape(P, K), y)
    delta_prob = (delta_yk.T - prob)
    grad_theta = np.concatenate([-np.sum(X - delta_prob[:, k][:, np.newaxis], axis=0) 
                    for k in range(prob.shape[1])])
    return grad_theta

from scipy.linalg import block_diag
def hess_softmax(theta, X, y):
    _, P = X.shape
    K = len(np.unique(y))
    _, _, _, prob = get_softmax_comps(X, theta.reshape(P, K), y)
    theta_mat = theta.reshape(P, K)
    H = block_diag(*[np.diag(theta_mat[:, k]) - prob[:, k].reshape(P, P) 
                 for k in range(K)])
    return H



"""
optimizing via scipy seems to only work with Nelder-Mead
Alternative is to use scikit-learn

"""
def optimize_softmax(X, y, method):
    N, P = X.shape
    M = len(np.unique(y)) # number of classes
    K = M # number of non redundant sets of parameters
    x0 = np.zeros(P*K)
    min = minimize(softmax_cost, x0, args=(X, y), jac=grad_softmax, hess=hess_softmax, method=method)
    return min



def fit_softmax_reg(X, y):
    model = linear_model.LogisticRegression(multi_class="multinomial", solver="newton-cg")
    model.fit(X, y)
    return model


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

n_samples = 3000
likelihood_dists = [ss.norm, ss.norm]
pi = [0.5, 0.5]
A = np.array([[0.6, 0.4],
              [0.2, 0.8]])
W = np.array([[10, 1], 
              [4, 0.5], 
              [15.5, 1.5]])

X = np.random.random((n_samples, 3))
y, Z = simulate_data_hmm_glm(n_samples, likelihood_dists, W, X, pi, A, [1.2, 0.3])


est, f, b, d = fit_hmm_glm_em(y, likelihood_dists, X)


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, 
                           n_redundant=0, n_classes=3, random_state=1)
# summarize the dataset
N, P = X.shape
K = 3
theta_mat = np.random.random((P, K))

test = optimize_softmax(X, y, "CG")
 


