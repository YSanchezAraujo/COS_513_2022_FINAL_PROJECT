mport hmm_glm

def fit_hmm_glm_em(y, 
              likelihood_dists, X,
              tol=1e-5, max_iter=50):

    N, K = np.shape(y)[0], len(likelihood_dists)

    # initialize with k-means
    init_params = init_kmeans(y, K)

    W_no_state = np.repeat(lm(X, y), K).reshape(X.shape[1], K)
    #for k in range(1, K):
    #    W_no_state[:, k] = W_no_state[:, k] + np.random.random(X.shape[1])

    # initialize parameters: pi, Phi, mu, sig2
    data = Data(np.zeros((N, K)), likelihood_dists, y, 
                      [i[1] for i in init_params], X, W_no_state)

    # initialize likelihood matrix
    data.compute_likelihoods()

    # if the current params result in all zero likelihood, initialize randomly
    for k in range(K):
        if data.lik[:, k].mean() == 0:
            data.lik[:, k] = np.random.random(N)


    # Q: do we actually need k-means initializations?
    forw_mesg = Forward_message(
        np.zeros((N, K)), _uniform_prob(K), 
        np.zeros(N), np.random.random((K, K))
    )

    back_mesg = Backward_message(np.zeros((N, K)))
    log_lik = 1e8
    ll = np.zeros(max_iter)

    for m in range(max_iter):
        # run forward message
        forw_mesg.run(data)

        # run backward message
        back_mesg.run(data, forw_mesg.Phi)

        # compute posteriors
        gamma, sig, pi, Phi, W  = posterior_estimates(forw_mesg, back_mesg, data)

        # compute log-likelihood
        log_lik_m = np.sum(np.log(forw_mesg.Z))
        log_lik_ch = np.abs(log_lik - log_lik_m)
        log_lik = log_lik_m
        print("LL change: ", log_lik_ch, "\t", "LL current: ", log_lik)
        ll[m] = log_lik

        data.W = W
        data.stdevs = sig

        # update likelihoods
        data.compute_likelihoods()

        # update forward object Phi
        forw_mesg.Phi = Phi
        forw_mesg.pi = pi

        # check convergence
        if log_lik_ch < tol:
            break

    # create final object to regurn
    posterior = {"W": W, "stdev": sig, "pi":pi, "Phi": Phi, "gamma": gamma}

    return posterior, forw_mesg, back_mesg, data, ll
    
