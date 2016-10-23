import numpy as np
import tensorflow as tf

### Base neural network                                                                                                                  
def init_mlp(layer_sizes, std=.01):
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
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_recon_linear, x), reduction_indices=1, keep_dims=True)


def gauss2gauss_KLD(mu_post, sigma_post, mu_prior, sigma_prior):
    d = (mu_post - mu_prior)
    d = tf.mul(d,d)
    return -.5 * tf.reduce_sum(-tf.div(d + tf.mul(sigma_post,sigma_post),sigma_prior*sigma_prior) \
                                    - 2*tf.log(sigma_prior) + 2.*tf.log(sigma_post) + 1., reduction_indices=1, keep_dims=True)


def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = tf.mul(-1., tf.mul(d,d))
    s2 = tf.mul(2., tf.mul(sigma,sigma))
    return tf.reduce_sum(tf.div(d2,s2) - tf.log(tf.mul(sigma, 2.506628)), reduction_indices=1, keep_dims=True)


### Gaussian Mixture Model VAE Class
class GaussVAE(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()


    def init_encoder(self, hyperParams):
        return {'base':init_mlp([hyperParams['input_d'], hyperParams['hidden_d']]), 
                'mu':init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]),
                'sigma':init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']])}


    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def f_prop(self):

        h1 = mlp(self.X, self.encoder_params['base'])
        
        self.mu = mlp(h1, self.encoder_params['mu'])
        self.sigma = tf.exp(mlp(h1, self.encoder_params['sigma']))
        self.z = self.mu + tf.mul(self.sigma, tf.random_normal(tf.shape(self.sigma)))
        x_recon_linear = mlp(self.z, self.decoder_params)

        return x_recon_linear


    def get_ELBO(self):
    
        # compose elbo
        elbo = -compute_nll(self.X, self.x_recons_linear)
        elbo -= gauss2gauss_KLD(self.mu, self.sigma, self.prior['mu'], self.prior['sigma'])

        return tf.reduce_mean(elbo)


    def get_log_margLL(self, batchSize):

        ll = -compute_nll(self.X, self.x_recons_linear)

        # calc prior
        log_prior = log_normal_pdf(self.z, self.prior['mu'], self.prior['sigma'])

        # calc post 
        log_post = log_normal_pdf(self.z, self.mu, self.sigma)

        return ll + log_prior - log_post


    def get_samples(self, nImages):
        z = self.prior['mu'] + tf.mul(self.prior['sigma'], tf.random_normal((nImages, tf.shape(self.decoder_params['w'][0])[0])))
        return tf.sigmoid(mlp(z, self.decoder_params))
