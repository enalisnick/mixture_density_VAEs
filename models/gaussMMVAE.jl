using AutoGrad
using Distributions
include("../utils/nn_utils.jl")
include("mixModel_grad_helper.jl")

function init_params(in_size, hidden_size, latent_size, n_comps, std=0.01)
  # encoder params
  w_encoder = append!([std*randn(in_size, hidden_size), std*randn(hidden_size, n_comps)], [std*randn(hidden_size, latent_size) for i=1:2*n_comps])
  b_encoder = append!([zeros(1, hidden_size), zeros(1, n_comps)], [zeros(1, latent_size) for i=1:2*n_comps])

  # decoder params
  w_decoder = Any[std*randn(latent_size, hidden_size), std*randn(hidden_size, in_size)]
  b_decoder = Any[zeros(1, hidden_size), zeros(1, in_size)]

  return Dict("w_encoder"=>w_encoder, "b_encoder"=>b_encoder, "w_decoder"=>w_decoder, "b_decoder"=>b_decoder)
end

# Splitting encoder and decoder fprops--will make grad easier
function fprop_encoder(params, x)
  n_components = size(params["w_encoder"][2], 2)
  latent_size = size(params["w_encoder"][3], 2)
  batchSize = size(x,1)

  h1 = relu(x*params["w_encoder"][1] .+ params["b_encoder"][1])

  # compute mixture parameters (all components -- needed for KL term)
  log_mix_weights = h1*params["w_encoder"][2] .+ params["b_encoder"][2]
  posterior = Dict("weights"=>softmax(log_mix_weights), "mu"=>[zeros(batchSize, latent_size) for i=1:n_components], "sigma"=>[zeros(batchSize, latent_size) for i=1:n_components])
  for k in 1:n_components
    posterior["mu"][k] += h1*params["w_encoder"][2*k+1] .+ params["b_encoder"][2*k+1]
    posterior["sigma"][k] += exp( h1*params["w_encoder"][2*k+2] .+ params["b_encoder"][2*k+2] )
  end

  # sample component index
  vals, idxs = findmax(log_mix_weights + rand(Gumbel(0,1), size(log_mix_weights)),2)
  comp_idxs = ind2sub(size(log_mix_weights), vec(idxs))[2]

  # sample latent variables from selected component distributions
  z = randn(batchSize, latent_size)
  for idx in 1:batchSize
    z[idx,:] = posterior["sigma"][comp_idxs[idx]][idx,:] .* z[idx,:] +  posterior["mu"][comp_idxs[idx]][idx,:]
  end

  return z, posterior
end

function fprop_decoder(params)

  h2 = relu(params["z"]*params["w_decoder"][1] .+ params["b_decoder"][1])
  x_recon = sigmoid(h2*params["w_decoder"][2] .+ params["b_decoder"][2])

  # clip the reconstruction to prevent NaN
  x_recon = min(max(x_recon, .000001), .999999)

  return x_recon
end

function fprop(params, x)
  z, posterior = fprop_encoder(params, x)
  x_recon = fprop_decoder(Dict("w_decoder"=>params["w_decoder"], "b_decoder"=>params["b_decoder"], "z"=>z))
  return x_recon, posterior, z
end




### KLD Terms ###
function gaussPDF(x, mu, sigma)
  return sum(1./sqrt(2*3.1416*sigma.^2) .* exp(-.5*(x - mu).^2 ./ sigma.^2), 2)
end

function gaussMixPDF(x, params)
  s = 0.
  for idx in 1:size(params["weights"],2)
    s += params["weights"][:,idx] .* gaussPDF(x, params["mu"][idx], params["sigma"][idx])
  end
  return s
end

function computeKLD(prior, posterior, x=None)
  # need MC approximation
  return log(gaussMixPDF(x, posterior))-log(gaussMixPDF(x, prior))
end



### Gradient Functions ###

function decoder_nll(params, x)
  # prop through decoder
  x_recon = fprop_decoder(params)

  # likelihood term
  expected_nll = sum(-x .* log(x_recon) - (1-x) .* log(1-x_recon), 2)

  return sum(expected_nll)
end

# get decoder grad via AutoGrad
nll_grad = grad(decoder_nll)

function elbo_grad(params, x, priorParams)
  # get dimensions
  in_size, hidden_size = size(params["w_encoder"][1])
  n_components = size(params["w_encoder"][2],2)
  latent_size = size(params["w_encoder"][3],2)

  # init gradient dictionary
  grads = init_params(in_size, hidden_size, latent_size, n_components, 0.)

  # get derivatives of ELBO wrt decoder params and z
  z, posterior = fprop_encoder(params, x)
  decoder_grads = nll_grad(Dict("w_decoder"=>params["w_decoder"], "b_decoder"=>params["b_decoder"], "z"=>z), x)



  return grads
end
