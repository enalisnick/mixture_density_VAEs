using AutoGrad
using Distributions
include("../utils/nn_utils.jl")
include("mixModel_grad_helper.jl")

function init_params(in_size, hidden_size, latent_size, n_comps, std=0.01)
  # encoder params
  w_encoder = [std*randn(in_size, hidden_size)] + [std*randn(hidden_size, n_comps)] + n_comps * [std*randn(hidden_size, latent_size), std*randn(hidden_size, latent_size)]
  b_encoder = [zeros(1, hidden_size)] + [zeros(1, n_comps)] + n_comps * [zeros(1, latent_size), zeros(1, latent_size)]

  # decoder params
  w_decoder = Any[std*randn(latent_size, hidden_size), std*randn(hidden_size, in_size)]
  b_decoder = Any[zeros(1, hidden_size), zeros(1, in_size)]

  return Dict("w_encoder"=>w_encoder, "b_encoder"=>b_encoder, "w_decoder"=>w_decoder, "b_decoder"=>b_decoder)
end

function fprop(params, x)
  n_components = size(params["w_encoder"][2], 2)
  latent_size = size(params["w_encoder"][3], 2)
  batchSize = size(x,1)

  # prop encoder
  h1 = relu(x*params["w_encoder"][1] .+ params["b_encoder"][1])

  # compute mixture parameters (all components -- needed for KL term)
  log_mix_weights = h1*params["w_encoder"][2] .+ params["b_encoder"][2]
  posterior = Dict("weights"=>softmax(log_mix_weights), "mu"=>[zeros(batchSize, latent_size)]*n_components, "sigma"=>[zeros(batchSize, latent_size)]*n_components)
  for k in 1:n_components
    posterior["mu"][k] += h1*params["w_encoder"][2*k+1] .+ params["b_encoder"][2*k+1]
    posterior["sigma"][k] += exp( h1*params["w_encoder"][2*k+2] .+ params["b_encoder"][2*k+2] )
  end

  # sample component index
  vals, comp_idxs = findmax(log_mix_weights + rand(Gumbel(0,1), size(log_mix_weights)))

  # sample latent variables from selected component distributions
  z = randn(batchSize, latent_size)
  for idx in 1:batchSize
    z[idx,:] = posterior["sigma"][comp_idxs[idx]][idx,:] .* z[idx,:] +  posterior["mu"][comp_idxs[idx]][idx,:]
  end

  # prop decoder
  h2 = relu(z*params["w_decoder"][1] .+ params["b_decoder"][1])
  x_recon = sigmoid(h2*params["w_decoder"][2] .+ params["b_decoder"][2])

  # clip the reconstruction to prevent NaN
  x_recon = min(max(x_recon, .000001), .999999)

  return x_recon, posterior, z
end

function gaussMixPDF(x, params)
  s = 0.
  for idx in len(params["weights"])
    s += params["weights"][idx] * gaussPDF(x, params["mu"][idx], params["sigma"][idx])
  end
  return s
end

function computeKLD(prior, posterior, x=None)
  # need MC approximation
  return log(gaussMixPDF(x, posterior))-log(gaussMixPDF(x, prior))
end

function custom_grad(objective)
  # TO DO
end
