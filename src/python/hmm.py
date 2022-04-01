import numpy as np
import scipy.stats as ss

def _normalize_z(x):
    Z = np.sum(x)
    return x / Z, Z

# tested, this works
class data:
    def __init__(self, lik, psi, y):
        self.lik = lik
        self.psi = psi
        self.y = y

    def update_likelihood_params(self, params):
        # figure out a way to update this without naming
        # i.e. rely on input ordre
        for k in range(len(self.psi)):
            self.psi[k] = self.psi[k](*params[k])

    def compute_likelihods(self):
        N = np.shape(self.y)[0]
        K = len(self.psi)

        for t in range(N):
            for k in range(K):
                self.lik[t, k] = self.psi[k].pdf(self.y[t])

class forward_message:
    def __init__(self, alpha, pi, Z, Phi):
        """
            alpha: sufficient statistics estimated from e step
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
            alpha_t, Z_t = _normalize_z(data.lik[t, :] * (self.Phi.T @ self.alpha[t-1, :]))
            self.alpha[t, :] = alpha_t
            self.Z[t] = Z_t
        

class backward_message:
    def __init__(self, beta, Phi):
        """
            beta: sufficient statistics
            Phi: Transition matrix
        """
        self.beta = beta
        self.Phi = Phi
    
    def run(self, data):
        N = np.shape(data.lik)[1]
        self.beta[N, :] = 1

        for t in range(N-1, 0, -1):
            beta_t, _ = _normalize_z(self.Phi @ (data.lik[t, :] * self.beta[t, :]))
            self.beta[t-1, :] = beta_t


def posterior_estimates(forward_message, backward_message, data):
    N = np.shape(data.lik)[0]
    K = len(data.psi)
    gamma = forward_message.alpha * backward_message.beta
    mu = np.zeros(K)

    for k in range(K):
        g = gamma[:, k]
        mu[k] = g[:, np.newaxis].T @ data.y[:, np.newaxis] / np.sum(g)
    
    sig2, zeta = np.zeros(K), np.zeros((K, K, N))
    y_no_mu = y - mu[:, np.newaxis]

    for k in range(K):
        g = gamma[:, k]
        sig2[k] = g[:, np.newaxis].T @ y_no_mu**2 / np.sum(gamma[:, k])

    pi = gamma[0, :] / np.sum(gamma[0, :])

    for t in range(N):
        lik_beta = data.lik[t+1, :] * backward_message.beta[t+1, :]
        alpha = forward_message.alpha[t, :]
        zeta[:, :, t] = (
            backward_message.Phi * (alpha[:, np.newaxis] @ lik_beta[:, np.newaxis].T)
        ) / forward_message.Z[t+1]

    zeta_nt = np.sum(zeta, axis=(2, 1))
    zeta = np.sum(zeta, axis=2) / zeta_nt

    return gamma, mu, np.sqrt(sig2), pi, zeta


# testing
N  = np.shape(dobj.lik)[0]
K = len(dobj.psi)
gamma = f.alpha * b.beta
mu = np.zeros(K)

for k in range(K):
    g = gamma[:, k]
    mu[k] = g[:, np.newaxis].T @ dobj.y / np.sum(g)

sig2, zeta = np.zeros(K), np.zeros((K, K, N))
