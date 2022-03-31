import numpy as np
import scipy.stats as ss

def _normalize_z(x):
    Z = np.sum(x)
    return x / Z, Z

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

    def run(self, data_obj):
        N = np.shape(data_obj.lik, 1)
        alpha_init, Z_init = _normalize_z(data_obj.lik[0, :] * self.pi)
        self.alpha[0, :] = alpha_init
        self.Z[0] = Z_init

        for t in range(1, N+1):
            alpha_t, Z_t = _normalize_z(data_obj.lik[t, :] * (self.Phi.T @ self.alpha[t-1, :]))
            self.alpha[t, :] = alpha_t
            self.Z[t] = Z_t
        

class backward_message:
    def __init(self, beta, Phi):
        """
            beta: sufficient statistics
            Phi: Transition matrix
        """
        self.beta = beta
        self.Phi = Phi
    
    def run(self, data_obj):
        N = np.shape(data_obj.lik, 1)
        self.beta[N, :] = 1

        for t in range(N-1, 0, -1):
            beta_t, _ = _normalize_z(self.Phi @ (data_obj.lik[t, :] * self.beta[t, :]))
            self.beta[t-1, :] = beta_t


def _posterior_estimates(forward_message, backward_message, data_obj):
    N = np.shape(data_obj.lik, 1)
    K = len(data_obj.psi)
    gamma = forward_message.alpha * forward_message.beta
    mu = np.zeros(K)

    for k in range(K):
        mu[k] = gamma[:, k].T @ data_obj.y / np.sum(gamma[:, k])
    
    sig2, zeta = np.zeros(K), np.zeros(K, K, N)
    y_no_mu = y - mu[:, np.newaxis]

    for k in range(K):
        sig2[k] = gamma[:, k].T @ y_no_mu**2 / np.sum(gamma[:, k])

    pi = gamma[0, :] / np.sum(gamma[0, :])

    for t in range(N):
        zeta[:, :, t] = (
            backward_message.Phi * (forward_message.alpha[t, :] * 
                                    (data_object.lik[t+1, :] * backward_message.beta[t+1, :].T))
        ) / forward_message.Z[t+1]

    zeta_nt = np.sum(zeta, axis=(2, 1))
    zeta = np.sum(zeta, axis=2) / zeta_nt

    return gamma, mu, np.sqrt(sig2), pi, zeta



