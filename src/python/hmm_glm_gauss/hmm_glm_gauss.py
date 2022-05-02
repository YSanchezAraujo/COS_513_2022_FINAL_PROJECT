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
    
    for k in range(K):
        p_z = np.diag(gamma[:, k] / np.sum(gamma[:, k]))
        X_proj = data.X.T @ p_z @ data.X
        y_proj = data.X.T @ p_z @ data.y
        W[:, k] = np.linalg.lstsq(X_proj, y_proj, rcond=None)[0]


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
