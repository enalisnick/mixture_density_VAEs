import os
from os.path import join as pjoin
import h5py
import cPickle as cp

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.gaussMMVAE_collapsed import GaussMMVAE
from utils.sampling_utils import *

try:
    import PIL.Image as Image
except ImportError:
    import Image


# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 100, "batch size.")
flags.DEFINE_integer("nEpochs", 100, "number of epochs to train.")
flags.DEFINE_float("adamLr", 3e-4, "AdaM learning rate.")
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
def trainVAE(data, vae_hyperParams, hyperParams, param_save_path, logFile=None):

    N_train, d = data['train'].shape
    N_valid, d = data['valid'].shape
    nTrainBatches = N_train/hyperParams['batchSize']
    nValidBatches = N_valid/hyperParams['batchSize']
    vae_hyperParams['batchSize'] = hyperParams['batchSize']

    # init Mix Density VAE
    model = GaussMMVAE(vae_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)

    # get op to save the model
    persister = tf.train.Saver()

    with tf.Session(config=hyperParams['tf_config']) as s:
        s.run(tf.initialize_all_variables())
        
        # for early stopping
        best_elbo = -10000000.
        best_epoch = 0

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
                persister.save(s, param_save_path)

            # log training progress
            logging_str = "Epoch %d.  Train ELBO: %.3f,  Validation ELBO: %.3f %s" %(epoch_idx+1, train_elbo, valid_elbo, star_printer)
            print logging_str
            if logFile: 
                logFile.write(logging_str + "\n")
                logFile.flush()

            # check for convergence
            if epoch_idx - best_epoch > hyperParams['lookahead_epochs']: break  

    return model



### Marginal Likelihood Calculation            
def calc_margLikelihood(data, model, param_file_path, vae_hyperParams, nSamples=50):
    N,d = data.shape

    # get op to load the model                                                                                               
    persister = tf.train.Saver()

    with tf.Session() as s:
        persister.restore(s, param_file_path)

        sample_collector = []
        for s_idx in xrange(nSamples):
            samples = s.run(model.get_log_margLL(N), {model.X: data})
            if not np.isnan(samples.mean()) and not np.isinf(samples.mean()):
                sample_collector.append(samples)
        
    if len(sample_collector) < 1:
        print "\tMARG LIKELIHOOD CALC: No valid samples were collected!"
        return np.nan

    all_samples = np.hstack(sample_collector)
    m = np.amax(all_samples, axis=1)
    mLL = m + np.log(np.mean(np.exp( all_samples - m[np.newaxis].T ), axis=1))
    return mLL.mean()


### Sample Images                                   
def sample_from_model(model, param_file_path, vae_hyperParams, image_file_path, nImages=100):

    # get op to load the model                                                                                                    
    persister = tf.train.Saver()

    with tf.Session() as s:
        persister.restore(s, param_file_path)
        sample_list = s.run(model.get_samples(nImages))

    for i, samples in enumerate(sample_list):
        image = Image.fromarray(tile_raster_images(X=samples, img_shape=(28, 28), tile_shape=(int(np.sqrt(nImages)), int(np.sqrt(nImages))), tile_spacing=(1, 1)))
        image.save(image_file_path+"_component"+str(i)+".png")


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

    
    # setup files to write results and save parameters
    outfile_base_name = get_file_name(inArgs.experimentDir, vae_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"train_logs/gaussMM_vae_trainResults"+outfile_base_name+".txt", 'w')
    param_file_name = inArgs.experimentDir+"params/gaussMM_vae_params"+outfile_base_name+".ckpt"

    # train
    print "Training model..."
    model = trainVAE(mnist, vae_hyperParams, train_hyperParams, param_file_name, logging_file)

    # evaluate marginal likelihood
    print "Calculating the marginal likelihood..."
    margll_valid = calc_margLikelihood(mnist['valid'], model, param_file_name, vae_hyperParams) 
    margll_test = calc_margLikelihood(mnist['test'], model, param_file_name, vae_hyperParams)
    logging_str = "\n\nValidation Marginal Likelihood: %.3f,  Test Marginal Likelihood: %.3f" %(margll_valid, margll_test)
    print logging_str
    logging_file.write(logging_str+"\n")
    logging_file.close()

    # draw some samples
    print "Drawing samples..."
    sample_from_model(model, param_file_name, vae_hyperParams, inArgs.experimentDir+'samples/gaussMM_vae_samples'+outfile_base_name)
