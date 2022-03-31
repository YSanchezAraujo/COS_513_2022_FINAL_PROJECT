import hmm

def fit_hmm_em(data, likelihood_dists, tol, max_iter):
    N, K = np.shape(data, 1) len(likelihood_dists)
    # initialize parameters: pi, Phi, mu, sig2

    for m in range(max_iter):
        # run forward message
        # run backward message
        # compute posteriors
        # compute log-likelihood
        # compute log-likelihood change
        # update likelihood distributon parameters
        # check convergence
    
