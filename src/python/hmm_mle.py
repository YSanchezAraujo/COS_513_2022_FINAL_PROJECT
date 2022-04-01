import hmm

def fit_hmm_em(y, likelihood_dists, tol, max_iter):
    N, K = np.shape(data, 1) len(likelihood_dists)
    # initialize parameters: pi, Phi, mu, sig2
    data_obj = data(np.zeros((N, K)), likelihood_dists, y)
    # need to get initial parameters
    
    # initialize likelihood matrix
    data_obj.compute_likelihoods()
    for m in range(max_iter):
        # run forward message
        # run backward message
        # compute posteriors
        # compute log-likelihood
        # compute log-likelihood change
        # update likelihood distributon parameters
        # check convergence
    
