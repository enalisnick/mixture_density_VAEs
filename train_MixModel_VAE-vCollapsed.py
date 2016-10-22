import os
from os.path import join as pjoin
import h5py
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
flags.DEFINE_string("experimentDir", "MNIST/", "directory to save training artifacts.")
inArgs = flags.FLAGS


def get_file_name(expDir, vaeParams, trainParams):     
    # concat hyperparameters into file name
    output_file_base_name = '_'+''.join('{}_{}_'.format(key, val) for key, val in sorted(vaeParams.items()) if key not in ['prior', 'input_d'])
    output_file_base_name += ''.join('{}_{}_'.format(key, "-".join([str(x) for x in vaeParams['prior'][key]])) for key in sorted(['mu', 'sigma']))
    output_file_base_name += 'dirichlet_alpha_'+str(vaeParams['prior']['dirichlet_alpha'])
    output_file_base_name += '_adamLR_'+str(trainParams['adamLr'])
                                                                               
    # check if results file already exists, if so, append a number                                                                                               
    results_file_name = pjoin(expDir, "train_logs/gaussMM_vae_trainResults"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(expDir, "train_logs/gaussMM_vae_trainResults"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)

    return output_file_base_name


### Training function
def trainVAE(data, vae_hyperParams, hyperParams, logFile=None):

    N_train, d = data['train'].shape
    N_valid, d = data['valid'].shape
    nTrainBatches = N_train/hyperParams['batchSize']
    nValidBatches = N_valid/hyperParams['batchSize']

    # init Mix Density VAE
    model = GaussMMVAE(vae_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)

    with tf.Session(config=hyperParams['tf_config']) as s:
        s.run(tf.initialize_all_variables())
        
        # for early stopping
        best_elbo = -10000000.
        best_epoch = 0
        encoder_params = None
        decoder_params = None

        for epoch_idx in xrange(hyperParams['nEpochs']):

            # training
            train_elbo = 0.
            for batch_idx in xrange(nTrainBatches):
                x = data['train'][batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
                _, elbo_val = s.run([optimizer, model.elbo_obj], {model.X: x})
                train_elbo += elbo_val

            # validation
            valid_elbo = 0.
            for batch_idx in xrange(nValidBatches):
                x = data['valid'][batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
                valid_elbo += s.run(model.elbo_obj, {model.X: x})

            # check for ELBO improvement
            star_printer = ""
            train_elbo /= nTrainBatches
            valid_elbo /= nValidBatches
            if valid_elbo > best_elbo: 
                best_elbo = valid_elbo
                best_epoch = epoch_idx
                star_printer = "***"
                # save the parameters                                                                                                                               
                encoder_params = {'mu':[], 'sigma':[]}
                decoder_params = {'w':[s.run(p) for p in model.decoder_params['w']], 'b':[s.run(p) for p in model.decoder_params['b']]}
                encoder_params['base'] = {'w':[s.run(p) for p in model.encoder_params['base']['w']], 'b':[s.run(p) for p in model.encoder_params['base']['b']]}
                encoder_params['kumar_a'] = {'w':[s.run(p) for p in model.encoder_params['kumar_a']['w']], 'b':[s.run(p) for p in model.encoder_params['kumar_a']['b']]}
                encoder_params['kumar_b'] = {'w':[s.run(p) for p in model.encoder_params['kumar_b']['w']], 'b':[s.run(p) for p in model.encoder_params['kumar_b']['b']]}
                for k in xrange(vae_hyperParams['K']):
                    encoder_params['mu'].append({'w':[s.run(p) for p in model.encoder_params['mu'][k]['w']], 'b':[s.run(p) for p in model.encoder_params['mu'][k]['b']]})
                    encoder_params['sigma'].append({'w':[s.run(p) for p in model.encoder_params['sigma'][k]['w']], 'b':[s.run(p) for p in model.encoder_params['sigma'][k]['b']]})

            # log training progress
            logging_str = "Epoch %d.  Train ELBO: %.3f,  Validation ELBO: %.3f %s" %(epoch_idx+1, train_elbo, valid_elbo, star_printer)
            print logging_str
            if logFile: 
                logFile.write(logging_str + "\n")
                logFile.flush()

            # check for convergence
            if epoch_idx - best_epoch > hyperParams['lookahead_epochs']: break  

    return encoder_params, decoder_params


if __name__ == "__main__":

    # load MNIST
    f = h5py.File('./MNIST/data/binarized_mnist.h5')
    mnist = {'train':np.copy(f['train']), 'valid':np.copy(f['valid']), 'test':np.copy(f['test'])}
    np.random.shuffle(mnist['train'])

    # set architecture params
    vae_hyperParams = {'input_d':mnist['train'].shape[1], 'hidden_d':inArgs.hidden_size, 'latent_d':inArgs.latent_size, 'K':inArgs.K, \
                           'prior':{'dirichlet_alpha':1., 'mu':[0.]*inArgs.K, 'sigma':[1.]*inArgs.K}}
    assert len(vae_hyperParams['prior']['mu']) == len(vae_hyperParams['prior']['sigma']) == vae_hyperParams['K']

    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize, 'lookahead_epochs':25, \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=True)}

    
    outfile_base_name = get_file_name(inArgs.experimentDir, vae_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"train_logs/gaussMM_vae_trainResults"+outfile_base_name+".txt", 'w')

    # train
    encParams, decParams = trainVAE(mnist, vae_hyperParams, train_hyperParams, logging_file)

    logging_file.close()

    # save params
    cp.dump([encParams, decParams], open(inArgs.experimentDir+'params/gaussMM_vae_params'+outfile_base_name+'.pkl','wb'), protocol=cp.HIGHEST_PROTOCOL)

    # evaluate marginal likelihood
