include("../utils/nn_utils.jl")

function init_params(in_size, hidden_size, latent_size, std=0.01)
  # encoder params
  w_encoder = [std*randn(in_size, hidden_size), std*randn(hidden_size, latent_size), std*randn(hidden_size, latent_size)]
  b_encoder = [zeros(1, hidden_size), zeros(1, latent_size), zeros(1, latent_size)]

  # decoder params
  w_decoder = Any[std*randn(latent_size, hidden_size), std*randn(hidden_size, in_size)]
  b_decoder = Any[zeros(1, hidden_size), zeros(1, in_size)]

  return Dict("w_encoder"=>w_encoder, "b_encoder"=>b_encoder, "w_decoder"=>w_decoder, "b_decoder"=>b_decoder)
end

function fprop(params, x)
  # prop encoder
  h1 = relu(x*params["w_encoder"][1] .+ params["b_encoder"][1])
  post_mu = h1*params["w_encoder"][2] .+ params["b_encoder"][2]
  post_log_sigma = h1*params["w_encoder"][3].+ params["b_encoder"][3]

  # sample latent variables
  z = post_mu + exp(post_log_sigma) .* randn(size(post_mu))

  # prop decoder
  h2 = relu(z*params["w_decoder"][1] .+ params["b_decoder"][1])
  x_recon = sigmoid(h2*params["w_decoder"][2] .+ params["b_decoder"][2])

  # clip the reconstruction to prevent NaN
  x_recon = min(max(x_recon, .000001), .999999)

  return x_recon, Dict("mu"=>post_mu, "sigma"=>exp(post_log_sigma))
end

function computeKLD(prior, posterior)
  kl = -2 *log(prior["sigma"])
  kl += -( posterior["sigma"].^2 + (posterior["mu"] - prior["mu"]).^2 / (prior["sigma"].^2) )
  kl += 2*log(posterior["sigma"]) + 1.
  return -.5*sum(kl,2)
end


