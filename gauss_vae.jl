using MNIST
using AutoGrad

function init_params(in_size, hidden_size, latent_size, std=0.0001)
  # encoder params
  w_encoder = [std*randn(in_size, hidden_size), std*randn(hidden_size, latent_size), std*randn(hidden_size, latent_size)]
  b_encoder = [zeros(hidden_size), zeros(latent_size), zeros(latent_size)]

  # decoder params
  w_decoder = Any[randn(latent_size, hidden_size), randn(hidden_size, in_size)]
  b_decoder = Any[zeros(hidden_size), zeros(in_size)]

  return Dict("w_encoder"=>w_encoder, "b_encoder"=>b_encoder, "w_decoder"=>w_decoder, "b_decoder"=>b_decoder)
end

function elbo(x, prior)

end

function trainGaussVAE(data, params, hyperParams)
  N,d = size(data)
  nBatches = N/hyperParams("batchSize")

  # get the derivatives via autograd
  elbo_grad = grad(elbo)

  for epoch_idx in 1:hyperParams["nEpochs"]
    elbo_tracker = 0.
    for batch_idx in 1:nBatches

      # compute elbo
      elbo_tracker += elbo(params, x, hyperParams["prior"])

      # get elbo gradients
      grads = elbo_grad(params, data[(batch_idx-1)*hyperParams("batchSize"):batch_idx*hyperParams("batchSize")], hyperParams["prior"])

      # update encoder
      for param_idx in 1:3
        params["w_encdoer"][param_idx] + hyperParams["lr"] * grads["w_encdoer"][param_idx]
        params["b_encdoer"][param_idx] + hyperParams["lr"] * grads["b_encdoer"][param_idx]
      end

      # update decoder
      for param_idx in 1:2
        params["w_decoder"][param_idx] + hyperParams["lr"] * grads["w_decoder"][param_idx]
        params["b_decoder"][param_idx] + hyperParams["lr"] * grads["b_decoder"][param_idx]
      end

    end
    println("Epoch %d. ELBO: %.3f", [epoch_idx, elbo_tracker/nBatches])
  end

  return params
end

function run_VAE()
  # load MNIST
  data = transpose(traindata()[1])

  # shuffle and reduce dataset
  shuffle(vec(data))
  data = data[1:10000,:]

  # set architecture parameters
  hidden_size = 500
  latent_size = 25
  vae_params = init_params(size(data,2), hidden_size, latent_size)

  # set hyperparams
  hyperParams = Dict("lr"=>0.01, "prior"=>Dict("mu"=>0., "sigma"=>1.), "nEpochs"=>50, "batchSize"=>100)
  final_vae_params = trainGaussVAE(data, vae_params, hyperParams)
end

run_VAE()


