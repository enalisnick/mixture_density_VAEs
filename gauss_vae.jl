using MNIST
using AutoGrad

function relu(x)
  return max(x, 0.)
end

function sigmoid(x)
  return 1./(1 + exp(-x))
end

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

  return x_recon, post_mu, exp(post_log_sigma)
end

function gaussKLD(prior_mu, prior_sigma, post_mu, post_sigma)
  kl = -2 *log(prior_sigma)
  kl += -( post_sigma.^2 + (post_mu - prior_mu).^2 / (prior_sigma.^2) )
  kl += 2*log(post_sigma) + 1.
  return -.5*sum(kl,2)
end

function elbo(params, x, prior)
  # prop through autoencoder
  x_recon, post_mu, post_sigma = fprop(params, x)

  # likelihood term
  expected_nll = sum(-x .* log(x_recon) - (1-x) .* log(1-x_recon), 2)

  # kl divergence term
  kld = gaussKLD(prior["mu"], prior["sigma"], post_mu, post_sigma)

  return mean(expected_nll + kld)
end

function adamUpdate(params, grads, adamParams, b1=.95, b2=.999, e=1e-8)
  adamParams["t"] += 1

  for paramType in ["w_encoder", "b_encoder", "w_decoder", "b_decoder"]
    if contains(paramType,"encoder") idx_limit = 3 else idx_limit = 2 end
    for param_idx in 1:idx_limit
      # mean term
      adamParams["m"][paramType][param_idx] = b1*adamParams["m"][paramType][param_idx] + (1-b1)*grads[paramType][param_idx]
      m_hat = adamParams["m"][paramType][param_idx] ./ (1-b1.^adamParams["t"])

      # variance term
      adamParams["v"][paramType][param_idx] = b2*adamParams["v"][paramType][param_idx] + (1-b2)*grads[paramType][param_idx].^2
      v_hat = adamParams["v"][paramType][param_idx] ./ (1-b2.^adamParams["t"])

      # update model param
      params[paramType][param_idx] -= adamParams["lr"] * m_hat./(sqrt(v_hat) + e)
    end
  end

  return params
end

function trainVAE(data, params, hyperParams)
  N,d = size(data)
  nBatches = div(N,hyperParams["batchSize"])

  # get the derivatives via autograd
  elbo_grad = grad(elbo)

  for epoch_idx in 1:hyperParams["nEpochs"]
    elbo_tracker = 0.
    for batch_idx in 1:nBatches

      # get minibatch
      x = data[(batch_idx-1)*hyperParams["batchSize"]+1:batch_idx*hyperParams["batchSize"],:]

      # compute elbo
      elbo_tracker += elbo(params, x, hyperParams["prior"])

      # get elbo gradients
      grads = elbo_grad(params, x, hyperParams["prior"])

      # perform AdaM update
      params = adamUpdate(params, grads, hyperParams["adamParams"])

    end
    @printf "Epoch %d. Neg. ELBO: %.3f \n" epoch_idx elbo_tracker/nBatches
  end

  return params
end

function run_VAE()
  # load MNIST (values in [0,255])
  data = transpose(traindata()[1])

  # shuffle, normalized, and reduce dataset
  shuffle(vec(data))
  data /= 255
  #data = data[1:10000,:]

  # set architecture parameters
  hidden_size = 500
  latent_size = 25
  vae_params = init_params(size(data,2), hidden_size, latent_size)

  # set hyperparams
  adamParams = Dict("lr"=>0.0001, "m"=>init_params(size(data,2), hidden_size, latent_size, 0.), "v"=>init_params(size(data,2), hidden_size, latent_size, 0.), "t"=>0)
  hyperParams = Dict("adamParams"=>adamParams, "prior"=>Dict("mu"=>0., "sigma"=>1.), "nEpochs"=>50, "batchSize"=>100)

  final_vae_params = trainVAE(data, vae_params, hyperParams)
end

run_VAE()
