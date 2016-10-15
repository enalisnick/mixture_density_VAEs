import numpy as np
import tensorflow as tf

### Base neural network                                                                                                                  
def init_mlp(layer_sizes, std=.1):
    params = {'w':[], 'b':[]}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(tf.Variable(tf.random_normal([n_in, n_out], stddev=std)))
        params['b'].append(tf.Variable(tf.zeros([n_out,])))
    return params

def mlp(X, params):
    h = [X]
    for w,b in zip(params['w'][:-1], params['b'][:-1]):
        h.append( tf.nn.relu( tf.matmul(h[-1], w) + b ) )
    return tf.matmul(h[-1], params['w'][-1]) + params['b'][-1]


### Gaussian Mixture Model VAE Class
class GaussMMVAE_collapsed(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons = self.f_prop(hyperParams)

        self.elbo_obj = self.get_ELBO()

    def f_prop(self, hyperParams):
        
        # init variational params
        self.mu = []
        self.sigma = []
        self.kumar_a = []
        self.kumar_b = []
        self.z = []
        x_recon = []

        h1 = mlp(self.X, self.encoder_params['base'])
        
        for k in xrange(self.K)
            self.mu.append(mlp(h1, self.encoder_params['mu'][k]))
            self.sigma.append(tf.exp(mlp(h1, self.encoder_params['sigma'][k])))
            self.kumar_a.append(tf.exp(mlp(h1, self.encoder_params['kumar_a'][k])))
            self.kumar_b.append(tf.exp(mlp(h1, self.encoder_params['kumar_b'][k])))
            self.z.append(self.mu[-1] + self.sigma[-1] * tf.random_normal(tf.shape(self.sigma[-1])))
            x_recon.append(self.z[-1], self.decoder_params)

        return x_recon

    def get_ELBO(self, epsilon=1e-8):
        nll = []
        for k in xrange(self.K):

            nll.append(tf.reduce_sum(self.X * tf.log(output_tensor + epsilon) - (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon)))
