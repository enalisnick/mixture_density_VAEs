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


def compute_nll(x, x_recon_linear):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_recon_linear, x), axis=1)



### Gaussian Mixture Model VAE Class
class GaussMMVAE_collapsed(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop(hyperParams)

        self.elbo_obj = self.get_ELBO()


    def init_encoder(self, hyperParams):
        return {'base':init_mlp([hyperParams['input_d'], hyperParams['hidden_d']]), 
                'mu':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'sigma':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'kumar_a':[init_mlp([hyperParams['hidden_d'], 1]) for k in xrange(self.K)],
                'kumar_b':[init_mlp([hyperParams['hidden_d'], 1]) for k in xrange(self.K)]}


    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def f_prop(self, hyperParams):

        # init variational params
        self.mu = []
        self.sigma = []
        self.kumar_a = []
        self.kumar_b = []
        self.z = []
        x_recon_linear = []

        h1 = mlp(self.X, self.encoder_params['base'])
        
        for k in xrange(self.K)
            self.mu.append(mlp(h1, self.encoder_params['mu'][k]))
            self.sigma.append(tf.exp(mlp(h1, self.encoder_params['sigma'][k])))
            self.kumar_a.append(tf.exp(mlp(h1, self.encoder_params['kumar_a'][k])))
            self.kumar_b.append(tf.exp(mlp(h1, self.encoder_params['kumar_b'][k])))
            self.z.append(self.mu[-1] + self.sigma[-1] * tf.random_normal(tf.shape(self.sigma[-1])))
            x_recon_linear.append(self.z[-1], self.decoder_params)

        return x_recon_linear


    def get_ELBO(self):
        logLike = []
        for k in xrange(self.K):
            logLike.append(compute_nll(self.X, self.x_recons_linear[k]))

        return elbo
