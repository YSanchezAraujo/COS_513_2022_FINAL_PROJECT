import hmm

def fit_hmm_em(y, 
              likelihood_dists,
              tol=1e-5, max_iter=50):

    N, K = np.shape(y)[0], len(likelihood_dists)

    # initialize with k-means
    theta = np.array([1/K for k in range(K)])

    # initialize parameters: pi, Phi, mu, sig2
    data = Data(np.zeros((N, K)), likelihood_dists, y, theta)

    # initialize likelihood matrix
    data.compute_likelihoods()

    # Q: do we actually need k-means initializations?
    forw_mesg = Forward_message(
        np.zeros((N, K)), _uniform_prob(K), 
        np.zeros(N), np.random.random((K, K))
    )

    back_mesg = Backward_message(np.zeros((N, K)))
    log_lik = 1e8
    Y = _indicator(Y, 2).T

    for m in range(max_iter):
        # run forward message
        forw_mesg.run(data)

        # run backward message
        back_mesg.run(data, forw_mesg.Phi)

        # compute posteriors
        gamma, pi, Phi  = posterior_estimates(forw_mesg, back_mesg, data)

        # M-step 
        # the problem is I'm thinking of emission probabilities in the wrong way here
        # really its just a (m by k) where m = 2 and k = number of states
        # have to update the likelihoods accordingly
        thetas = (gamma.T @ Y) / np.sum(gamma, axis=0, keepdims=True)
        
        # compute log-likelihood
        log_lik_m = np.sum(np.log(forw_mesg.Z))
        log_lik_ch = np.abs(log_lik - log_lik_m)
        log_lik = log_lik_m

        # update likelihood distributon parameters
        data.compute_likelihoods()

        # update forward object Phi
        forw_mesg.Phi = Phi
        forw_mesg.pi = pi

        # check convergence
        if log_lik_ch < tol:
            break

    # create final object to regurn
    posterior = {"theta": data.theta, "pi":pi, "Phi": Phi, "gamma": gamma}

    return posterior, forw_mesg, back_mesg, data
    
