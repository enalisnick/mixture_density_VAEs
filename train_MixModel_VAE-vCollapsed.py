import os
import cPickle as cp
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.gaussMMVAE_collapsed import GaussMMVAE

# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 100, "batch size.")
flags.DEFINE_integer("nEpochs", 300, "number of epochs to train.")
flags.DEFINE_float("adamLr", 1e-4, "AdaM learning rate.")
flags.DEFINE_integer("hidden_size", 500, "number of hidden units in en/decoder.")
flags.DEFINE_integer("latent_size", 5, "dimensionality of latent variables.")
flags.DEFINE_integer("K", 5, "number of components in mixture model.")
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
            pi_tracker = [0.]*vae_hyperParams['K']
            for batch_idx in xrange(nBatches):
                
                # get minibatch
                x = data[batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
            
                # perform update
                _, elbo_val, pi = s.run([optimizer, model.elbo_obj, model.pi_samples], {model.X: x})
                
                for k in xrange(vae_hyperParams['K']): pi_tracker[k] += pi[k].sum()
                elbo_tracker += elbo_val

            print "Epoch %d.  ELBO: %.3f" %(epoch_idx, elbo_tracker/nBatches)
            print [pi_tracker[k]/(nBatches*hyperParams['batchSize']) for k in xrange(vae_hyperParams['K'])]
            print
        
        # save the parameters
        encoder_params = {'mu':[], 'sigma':[]}
        encoder_params['base'] = {'w':[s.run(p) for p in model.encoder_params['base']['w']], 'b':[s.run(p) for p in model.encoder_params['base']['b']]}
        encoder_params['kumar_a'] = {'w':[s.run(p) for p in model.encoder_params['kumar_a']['w']], 'b':[s.run(p) for p in model.encoder_params['kumar_a']['b']]}
        encoder_params['kumar_b'] = {'w':[s.run(p) for p in model.encoder_params['kumar_b']['w']], 'b':[s.run(p) for p in model.encoder_params['kumar_b']['b']]}
        for k in xrange(vae_hyperParams['K']): 
            encoder_params['mu'].append({'w':[s.run(p) for p in model.encoder_params['mu'][k]['w']], 'b':[s.run(p) for p in model.encoder_params['mu'][k]['b']]})
            encoder_params['sigma'].append({'w':[s.run(p) for p in model.encoder_params['sigma'][k]['w']], 'b':[s.run(p) for p in model.encoder_params['sigma'][k]['b']]})
        decoder_params = {'w':[s.run(p) for p in model.decoder_params['w']], 'b':[s.run(p) for p in model.decoder_params['b']]}
    
    return encoder_params, decoder_params


if __name__ == "__main__":

    # load MNIST
    mnist = input_data.read_data_sets("./MNIST/", one_hot=False)[0].images
    
    # shuffle and reduce
    np.random.shuffle(mnist)
    mnist = mnist[:45000,:]

    # set architecture params
    vae_hyperParams = {'input_d':mnist.shape[1], 'hidden_d':inArgs.hidden_size, 'latent_d':inArgs.latent_size, 'K':inArgs.K, \
                           'prior':{'dirichlet_alpha':1., 'mu':[0.]*inArgs.K, 'sigma':[1.]*inArgs.K}}
    assert len(vae_hyperParams['prior']['mu']) == len(vae_hyperParams['prior']['sigma']) == vae_hyperParams['K']

    # set hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize}

    encParams, decParams = trainVAE(mnist, vae_hyperParams, train_hyperParams)
    cp.dump([encParams, decParams], open('mixVAE_params.pkl','wb'), protocol=cp.HIGHEST_PROTOCOL)
