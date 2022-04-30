import hmm_glm_bern

def fit_hmm_glm_em(y, 
              likelihood_dists, X,
              tol=1e-5, max_iter=50):

    N, K = np.shape(y)[0], len(likelihood_dists)
    P = X.shape[1]

    W_no_state = np.random.random((P, K))

    # initialize parameters: pi, Phi, mu, sig2
    data = Data(np.zeros((N, K)), likelihood_dists, y, X, W_no_state)

    # initialize likelihood matrix
    data.compute_likelihoods()

    # Q: do we actually need k-means initializations?
    forw_mesg = Forward_message(
        np.zeros((N, K)), _uniform_prob(K), 
        np.zeros(N), np.random.random((K, K))
    )

    back_mesg = Backward_message(np.zeros((N, K)))
    log_lik = 1e8

    for m in range(max_iter):
        # run forward message
        forw_mesg.run(data)

        # run backward message
        back_mesg.run(data, forw_mesg.Phi)

        # compute posteriors
        gamma, pi, Phi = posterior_estimates(forw_mesg, back_mesg, data)

        # compute log-likelihood
        log_lik_m = np.sum(np.log(forw_mesg.Z))
        log_lik_ch = np.abs(log_lik - log_lik_m)
        log_lik = log_lik_m
  
        # now M-step
        for k in range(K):
            opt = optimize_logistic(data.X, data.y, gamma[:, k][:, np.newaxis], "SLSQP")
            data.W[:, k] = opt.x

        # update likelihoods
        data.compute_likelihoods()

        # update forward object Phi
        forw_mesg.Phi = Phi
        forw_mesg.pi = pi

        # check convergence
        if log_lik_ch < tol:
            break

    # create final object to regurn
    posterior = {"W": data.W,  "pi":pi, "Phi": Phi, "gamma": gamma}

    return posterior, forw_mesg, back_mesg, data
    
