include("../utils/utils.jl")
include("../models/vanilla_hmm.jl")
using Clustering

"""
inputs: \n
data: 1-d array of continuous values \n
dists: 1-d array of k Distribution objects w/o any parameters
      example: [Normal, Normal]
      if the number of latent variables being fit to the data is 2
"""
function fit_hmm_em(data, dists; tol=1e-4, max_iter=250)
    if size(data, 2) > 1
        throw(DomainError("n-dimensional samples not supported"))
    end
    T, K = size(data, 1), length(dists)
    # using k-means as an first guess
    init_pi, dist_params, A, init_mu, init_sd = init_estimates_kmeans(y, K, max_iter)
    # construct poseterior object       
    pos = posterior_object(init_mu, init_sd, zeros(T, K), init_pi, A)
    ll, ll_change = 0, 1
    dm = data_models(data, dists, zeros(T, K))
    compute_likelihoods!(dm, dist_params)
    f_msg = forward_object(zeros(T, K), init_pi, A, zeros(T))
    b_msg = backward_object(zeros(T, K), A) 
    for m in 1:max_iter
        if ll_change < tol
            println("converged in: ", m, " iterations")
            break
        end
        # run messages forward
        forward_message!(f_msg, dm)
        # run messages backwards
        backward_message!(b_msg, dm)
        # compute posterior quantities
        pos = compute_posteriors(f_msg, b_msg, dm)
        # get current log-likelihood
        ll_cur = sum(log.(f_msg.Z))
        ll_change = abs.(ll - ll_cur)
        ll = ll_cur
        dist_params = [(pos.μ[k], pos.σ[k]) for k in 1:K]
        compute_likelihoods!(dm, dist_params)
        update_forward_message!(f_msg, pos)
        update_backward_message!(b_msg, pos)
    end
    return pos, f_msg, b_msg, dm, ll
end

function viterbi(pos, dm)
    T, K = size(dm.L, 1), length(pos.π)
    z_max, traceback = zeros(Int64, T), zeros(T, K)
    δ, ll = zeros(T, K), 0
    δ[1, :] = dm.L[1, :] .* pos.π
    for t in 2:T
        δt = dm.L[t, :] .* (pos.Ψ' * δ[t-1, :])
        # todo
    end
    return z_max, traceback
end

