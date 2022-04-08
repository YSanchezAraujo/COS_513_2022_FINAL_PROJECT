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


# I think we only need to compute this once for each iteration of m step
def _softmax_normalizer(x, theta_mat):
    theta_x = theta_mat.T @ x
    return np.sum(np.exp(theta_x))

def _log_softmax(x, theta_mat, k):
    e_theta_x = _softmax_normalizer(theta_mat, x)
    return np.dot(x, theta_mat[:, k]) - np.log(e_theta_x)

def _indicator_y(y, k):
    return (y == k).astype(int)

def _indicator_Y(y, k_max):
    return np.vstack([_indicator_y(y, k) for k in range(k_max)])

def softmax_cost_per_sample(x, theta_mat, y):
    P, K = theta_mat.shape
    assert P == len(x)
    log_sm = np.array([_log_softmax(x, theta_mat, k) for k in range(K)])
    #log_SM = np.tile(log_sm, (K, 1)) # K by K matrix
    delta_YK = _indicator_Y(y, K) # K by M matrix in {0, 1}
    return -np.sum(delta_YK.T @ log_sm)


# this is much slower than list comprehension, just do that
# def softmax_total_cost(X, theta_mat, y, axis):
#     return np.apply_along_axis(softmax_cost_per_sample, axis, 
#                                arr=X, theta_mat=theta_mat, y=y).sum()
