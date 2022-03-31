function normalize_z(x)
    Z = sum(x)
    return x ./ Z,  Z
end

normalize_1o(x) = x ./ sum(x)
add_dim(x) = reshape(x, (size(x)..., 1))
drop_dim(a) = dropdims(a, dims = (findall(size(a) .== 1)...,))

"""
params here is just a list of variances
"""
function make_fake_data_glmhmm_stationary(n_samples, obs_dists, params, init_p, A, X, W)
    Z = zeros(Int64, n_samples)
    y = zeros(Float64, n_samples)
    # for redundancy
    Z[1] = rand(Categorical(init_p))
    mu1sig1 = [X[1, :]'W[:, Z[1]], params[Z[1]]]
    y[1] = rand(obs_dists[Z[1]](mu1sig1...))
    muSigma = zeros(length(y), 2)
    muSigma[1, :] = mu1sig1
    for t in 2:n_samples
        Z[t] = rand(Categorical(A[Z[t-1], :]))
        μ = X[t, :]'W[:, Z[t]]
        params_t = [μ, params[Z[t]]]
        muSigma[t, :] = params_t
        y[t] = rand(obs_dists[Z[t]](params_t...))
    end
    return y, Z, A, muSigma
end

"""
params here is a list of tuples that has mean and variance
"""
function make_fake_data_glmhmm(n_samples, obs_dists, params, X, W)
    Z = zeros(Int64, n_samples)
    y = zeros(Float64, n_samples)
    A = mapslices(softmax, X*W, dims=2)
    # for redundancy
    Z[1] = rand(Categorical(A[1, :]))
    y[1] = rand(obs_dists[Z[1]](params[Z[1]]...))
    for t in 2:n_samples
        Z[t] = rand(Categorical(A[t, :]))
        y[t] = rand(obs_dists[Z[t]](params[Z[t]]...))
    end
    return y, Z, A
end

"""
params here is a list of tuples that has mean and variance
"""
function make_fake_data_hmm(n_samples, obs_dists,  params, init_p, A)
    Z = zeros(Int64, n_samples)
    y = zeros(Float64, n_samples)
    Z[1] = rand(Categorical(init_p))
    y[1] = rand(obs_dists[Z[1]](params[Z[1]]...))
    for t in 2:n_samples
        Z[t] = rand(Categorical(A[Z[t-1], :]))
        y[t] = rand(obs_dists[Z[t]](params[Z[t]]...))
    end
    return y, Z
end