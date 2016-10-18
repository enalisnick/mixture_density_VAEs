import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.gaussMMVAE_collapsed import GaussMMVAE

# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 100, "batch size.")
flags.DEFINE_integer("nEpochs", 100, "number of epochs to train.")
flags.DEFINE_float("adamLr", 1e-4, "AdaM learning rate.")
flags.DEFINE_integer("hidden_size", 500, "number of hidden units in en/decoder.")
flags.DEFINE_integer("latent_size", 25, "dimensionality of latent variables.")
flags.DEFINE_integer("K", 2, "number of components in mixture model.")
inArgs = flags.FLAGS


### Training function
def trainVAE(data, vae_hyperParams, hyperParams):
    N,d = data.shape
    nBatches = N/hyperParams['batchSize']

    # init Mix Density VAE
    model = GaussMMVAE(vae_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)
    
    # train
    with tf.Session() as s:
        s.run(tf.initialize_all_variables())
        for epoch_idx in xrange(hyperParams['nEpochs']):
            elbo_tracker = 0.
            for batch_idx in xrange(nBatches):
                
                # get minibatch
                x = data[batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
            
                # perform update
                _, elbo_val = s.run([optimizer, model.elbo_obj], {model.X: x})
                
                elbo_tracker += elbo_val

            print "Epoch %d.  ELBO: %.3f" %(epoch_idx, elbo_tracker/nBatches)
        
        # save the parameters
        #encoder_params = [s.run(p) for p in encoder_params['w']] + [s.run(p) for p in encoder_params['b']]
        #decoder_params = [s.run(p) for p in decoder_params['w']] + [s.run(p) for p in decoder_params['b']]
    
    return None, None #encoder_params, decoder_params


if __name__ == "__main__":

    # load MNIST
    mnist = input_data.read_data_sets("./MNIST/", one_hot=False)[0].images

    # set architecture params
    vae_hyperParams = {'input_d':mnist.shape[1], 'hidden_d':inArgs.hidden_size, 'latent_d':inArgs.latent_size, 'K':inArgs.K, 'prior':{'dirichlet_alpha':1., 'mu':[-2., 2.], 'sigma':[1., 1.]}}
    assert len(vae_hyperParams['prior']['mu']) == len(vae_hyperParams['prior']['sigma']) == vae_hyperParams['K']

    # set hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize}

    encParams, decParams = trainVAE(mnist, vae_hyperParams, train_hyperParams)
    
