import utils

# tested, this works
class Data:
    def __init__(self, lik, psi, y, stdevs, X, W):
        self.lik = lik
        self.psi = psi
        self.y = y
        self.stdevs = stdevs
        self.X = X
        self.W = W

    def compute_likelihoods(self):
        N = np.shape(self.y)[0]
        K = len(self.psi)
        
        XW = self.X @ self.W 
        for k in range(K):
            self.lik[:, k] = self.psi[k](XW[:, k], self.stdevs[k]).pdf(self.y)

class Forward_message:
    def __init__(self, alpha, pi, Z, Phi):
        """
            pi: initial distribution
            Z: normalizing constant
            Phi: transition matrix
        """ 
        self.alpha = alpha
        self.pi = pi
        self.Z = Z
        self.Phi = Phi

    def run(self, data):
        N = np.shape(data.lik)[0]
        alpha_init, Z_init = _normalize_z(data.lik[0, :] * self.pi)
        self.alpha[0, :] = alpha_init
        self.Z[0] = Z_init

        for t in range(1, N):
            alpha_t, Z_t = _normalize_z(data.lik[t, :][:, np.newaxis] * 
                                        (self.Phi.T @ self.alpha[t-1, :][:, np.newaxis]))
            self.alpha[t, :] = alpha_t.ravel()
            self.Z[t] = Z_t
        

class Backward_message:
    def __init__(self, beta):
        self.beta = beta
    
    def run(self, data, Phi):
        N = np.shape(data.lik)[0]
        self.beta[N-1, :] = 1

        for t in range(N-1, 0, -1):
            beta_t, _ = _normalize_z(Phi @ (data.lik[t, :] * self.beta[t, :])[:, np.newaxis])
            self.beta[t-1, :] = beta_t.ravel()

# problem areas to solve are here
def posterior_estimates(forward_message, backward_message, data):
    N = np.shape(data.lik)[0]
    K = len(data.psi)
    gamma = forward_message.alpha * backward_message.beta
    W = np.zeros((np.shape(data.X)[1], K))
    X_proj = np.linalg.inv(data.X.T @ data.X) @ data.X.T
    p_z = gamma / gamma.sum(0)[np.newaxis, :]
    
    for k in range(K):
        W[:, k] = X_proj * p_z[:, k].T @ data.y * N

    sig2, zeta = np.zeros(K), np.zeros((N, K, K))
    y_no_mu = (data.y[:, np.newaxis] - data.X @ W)

    for k in range(K):
        g = gamma[:, k]
        sig2[k] = ((g[:, np.newaxis].T @ y_no_mu[:, k]**2 )/ np.sum(gamma[:, k]))

    pi = gamma[0, :] / np.sum(gamma[0, :])

    for t in range(N-1):
        lik_beta = data.lik[t+1, :] * backward_message.beta[t+1, :]
        alpha = forward_message.alpha[t, :]
        zeta[t, :, :] = (
            forward_message.Phi * (alpha[:, np.newaxis] @ lik_beta[:, np.newaxis].T)
        ) / forward_message.Z[t+1]

    Phi = np.sum(zeta, axis=0)
    Phi = Phi / Phi.sum(axis=1)[:, np.newaxis]

    return gamma, np.sqrt(sig2), pi, Phi, W



def _indicator_y(y, k):
    return (y == k).astype(int)


def _indicator_Y(y, k_max):
    return np.vstack([_indicator_y(y, k) for k in range(k_max)])


def get_softmax_comps(X, theta_mat, y):
    _, P = X.shape
    P2, K = theta_mat.shape
    assert P == P2
    x_dot_theta = X @ theta_mat
    # 709 is float64 precision
    exp_x_dot_theta = np.exp(np.clip(x_dot_theta, -709.78, 709.78))
    exp_x_dot_theta_norm = np.sum(exp_x_dot_theta, axis=1)
    prob = exp_x_dot_theta / exp_x_dot_theta_norm[:, np.newaxis]
    # log normalizer
    log_norm = np.log(exp_x_dot_theta_norm)
    log_prob = x_dot_theta - log_norm[:, np.newaxis]
    delta_yk = _indicator_Y(y, K)
    cost = -np.sum(delta_yk * log_prob.T)
    return cost, delta_yk, log_prob, prob


def softmax_cost(theta, X, y):
    N, P = X.shape
    K = len(np.unique(y)) - 1
    cost, _, _, _ = get_softmax_comps(X, theta.reshape(P, K), y)
    return cost


def grad_softmax(theta, X, y):
    N, P = X.shape
    K = len(np.unique(y)) - 1
    _, delta_yk, _, prob = get_softmax_comps(X, theta.reshape(P, K), y)
    delta_prob = (delta_yk.T - prob)
    grad_theta = np.concatenate([np.sum(X - delta_prob[:, k][:, np.newaxis], axis=0) 
                    for k in range(prob.shape[1])])
    return grad_theta



def optimize_softmax(X, y, method):
    N, P = X.shape
    M = len(np.unique(y)) # number of classes
    K = M - 1 # number of non redundant sets of parameters
    x0 = np.ones(P * K) * 0.1
    min = minimize(softmax_cost, x0, args=(X, y), jac=grad_softmax, method=method)
    return min

# this is much slower than list comprehension, just do that
# def softmax_total_cost(X, theta_mat, y, axis):
#     return np.apply_along_axis(softmax_cost_per_sample, axis, 
# 
# 
#                                arr=X, theta_mat=theta_mat, y=y).sum()
# for testing, optimization currently doesnt work
from collections import Counter
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# summarize the dataset
N, P = X.shape
K = 3
theta_mat = np.random.random((P, K))

cost, delta_yk, log_prob, prob = get_softmax_comps(X, theta_mat, y)
grad_softmax(delta_yk, prob, X)
