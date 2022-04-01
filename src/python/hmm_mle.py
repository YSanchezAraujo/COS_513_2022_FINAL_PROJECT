import hmm

def fit_hmm_em(y, 
              likelihood_dists,
              tol=1e-5, max_iter=50):

    N, K = np.shape(y)[0], len(likelihood_dists)

    # initialize with k-means
    init_params = init_kmeans(y, K)

    # initialize parameters: pi, Phi, mu, sig2
    data = Data(np.zeros((N, K)), likelihood_dists, y, init_params)

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
        gamma, mu, sig, pi, zeta = posterior_estimates(forw_mesg, back_mesg, data)

        # compute log-likelihood
        log_lik_m = np.sum(np.log(forw_mesg.Z))
        log_lik_ch = np.abs(log_lik - log_lik_m)
        log_lik = log_lik_m

        # update likelihood distributon parameters
        likelihood_params = [(mu[k], sig[k]) for k in range(K)]
        data.update_likelihood_params(likelihood_params)
        data.compute_likelihoods()

        # update forward object Phi
        forw_mesg.Phi = zeta
        forw_mesg.pi = pi

        # check convergence
        if log_lik_ch < tol:
            break

    # create final object to regurn
    posterior = {"mean": mu, "stdev": sig, "pi":pi, "zeta": zeta, "gamma": gamma}

    return posterior, forw_mesg, back_mesg, data

    
