import os
from os.path import join as pjoin
import h5py
import cPickle as cp

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.gaussMMVAE_collapsed import GaussMMVAE
from utils.sampling_utils import *
from train_MixModel_VAE_vCollapsed import *

try:
    import PIL.Image as Image
except ImportError:
    import Image


# command line arguments
# flags = tf.flags
# flags.DEFINE_integer("batchSize", 100, "batch size.")
# flags.DEFINE_integer("nEpochs", 500, "number of epochs to train.")
# flags.DEFINE_float("adamLr", 3e-4, "AdaM learning rate.")
# flags.DEFINE_integer("hidden_size", 500, "number of hidden units in en/decoder.")
# flags.DEFINE_integer("latent_size", 5, "dimensionality of latent variables.")
# flags.DEFINE_integer("K", 5, "number of components in mixture model.")
# flags.DEFINE_string("experimentDir", "MNIST/", "directory to save training artifacts.")
# inArgs = flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def get_embedding_samples(data, model, param_file_path, vae_hyperParams, hyperParams, nSamples=50):
    # get op to save the model
    persister = tf.train.Saver()

    # with tf.device('/gpu:3'):
    with tf.Session(config=hyperParams['tf_config']) as s:
        persister.restore(s, param_file_path)

        z_samples = np.zeros((data.shape[0], vae_hyperParams['latent_d']))

        # validation
        valid_elbo = 0.
        for batch_idx in xrange(1):
            x = data[batch_idx*100:(batch_idx+1)*100,:]
            z_samples = s.run([model.get_component_samples(vae_hyperParams['latent_d'], 100)], {model.X: x})

        # unif_sample = np.random.uniform(size=(100, 4))
        # kumar_sample = (1. - (1. - unif_sample)**(1/b))**(1/a)
        # weight_sample = np.hstack([weight_sample, (1.-weight_sample.sum(axis=1)).reshape(-1,1)])
        # component_sample = weight_sample.argmax(axis=1)

    return z_samples


def get_knn_predictions(train_data, test_data, model, param_file_path, vae_hyperParams, nSamples=50):
    import scipy as scp
    import scipy.special
    import sklearn as skl
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression


    train_embeddings = get_embedding_samples(train_data, model, param_file_path, nSamples=1)


    # then fit knn to samples
    print "fitting classifier..."
    lr = KNeighborsClassifier(n_neighbors = 3) #LogisticRegression(multi_class='multinomial', solver='newton-cg')
    lr.fit(train_embeddings, train_set_y.ravel())

    # then get samples for test
    test_embeddings = get_embedding_samples(test_data, model, param_file_path, nSamples=1)

    # then classify test based on knn classifier
    y_hats = lr.predict(samples)




    return mLL.mean()




if __name__ == "__main__":

    # load MNIST
    f = h5py.File('./MNIST/data/binarized_mnist.h5')
    mnist = {'train':np.copy(f['train']), 'valid':np.copy(f['valid']), 'test':np.copy(f['test'])}
    np.random.shuffle(mnist['train'])

    # set architecture params
    vae_hyperParams = {'input_d':mnist['train'].shape[1], 'hidden_d':inArgs.hidden_size, 'latent_d':inArgs.latent_size, 'K':inArgs.K, \
                           'prior':{'dirichlet_alpha':1., 'mu':[-0.5, -.2, 0., .2, 0.5], 'sigma':[1.]*inArgs.K}}
    #'prior':{'dirichlet_alpha':1., 'mu':[0.]*inArgs.K, 'sigma':[1.]*inArgs.K}}
    vae_hyperParams['K']=5
    assert len(vae_hyperParams['prior']['mu']) == len(vae_hyperParams['prior']['sigma']) == vae_hyperParams['K']

    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize, 'lookahead_epochs':25, \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=False)}



    # setup files to write results and save parameters
    # outfile_base_name = get_file_name(inArgs.experimentDir, vae_hyperParams, train_hyperParams)
    outfile_base_name = "_K_5_hidden_d_500_latent_d_5_mu_-0.5--0.2-0.0-0.2-0.5_sigma_1.0-1.0-1.0-1.0-1.0_dirichlet_alpha_1.0_adamLR_0.0003_25"
    logging_file = open(inArgs.experimentDir+"train_logs/gaussMM_vae_trainResults"+outfile_base_name+".txt", 'w')
    param_file_name = inArgs.experimentDir+"params/gaussMM_vae_params"+outfile_base_name+".ckpt"
    # param_file_name = "./MNIST/params/gaussMM_vae_params_K_3_hidden_d_500_latent_d_25_mu_0.0-0.0-0.0_sigma_1.0-1.0-1.0_dirichlet_alpha_1.0_adamLR_0.0003.ckpt"

    # train
    print "Load model..."
    model = trainVAE(mnist, vae_hyperParams, train_hyperParams, param_file_name, logging_file)


    z_samples = get_embedding_samples(mnist['valid'], model, param_file_name, vae_hyperParams, hyperParams=train_hyperParams)

    print z_samples
    print z_samples[0].shape
    print mnist['valid'].shape

    # # evaluate marginal likelihood
    # print "Calculating the marginal likelihood..."
    # margll_valid = calc_margLikelihood(mnist['valid'], model, param_file_name, vae_hyperParams)
    # margll_test = calc_margLikelihood(mnist['test'], model, param_file_name, vae_hyperParams)
    # logging_str = "\n\nValidation Marginal Likelihood: %.3f,  Test Marginal Likelihood: %.3f" %(margll_valid, margll_test)
    # print logging_str
    # logging_file.write(logging_str+"\n")
    # logging_file.close()

    # # draw some samples
    # print "Drawing samples..."
    # sample_from_model(model, param_file_name, vae_hyperParams, inArgs.experimentDir+'samples/gaussMM_vae_samples'+outfile_base_name,  inArgs.nEpochs)
