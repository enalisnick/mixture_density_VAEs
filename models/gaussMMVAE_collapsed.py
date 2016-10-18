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
    return tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(x_recon_linear, x), reduction_indices=1, keep_dims=True)


def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    d = (mu_post - mu_prior)
    d = tf.mul(d,d)
    return tf.reduce_sum(-tf.div(d + tf.mul(sigma_post,sigma_post),(2.*sigma_prior*sigma_prior)) - tf.log(sigma_prior*2.506628), reduction_indices=1, keep_dims=True)


def beta_fn(a,b):
    return tf.exp( tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a+b) )


def compute_kumar2beta_kld(a, b, alpha, beta):
    # precompute some terms
    ab = tf.mul(a,b)
    a_inv = tf.pow(a, -1)
    b_inv = tf.pow(b, -1)

    # compute taylor expansion for E[log (1-v)] term
    kl = tf.mul(tf.pow(1+ab,-1), beta_fn(a_inv, b))
    for idx in xrange(10):
        kl += tf.mul(tf.pow(idx+2+ab,-1), beta_fn(tf.mul(idx+2., a_inv), b))
    kl = tf.mul(tf.mul(beta-1,b), kl)

    kl += tf.mul(tf.div(a-alpha,a), -0.57721 - tf.digamma(b) - b_inv)
    # add normalization constants                                                                                                                         
    kl += tf.log(ab) + tf.log(beta_fn(alpha, beta))

    # final term                                                                                                  
    kl += tf.div(-(b-1),b)

    return kl


def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = tf.mul(-1., tf.mul(d,d))
    s2 = tf.mul(2., tf.mul(sigma,sigma))
    return tf.reduce_sum(tf.div(d2,s2) - tf.log(tf.mul(sigma, 2.506628)), reduction_indices=1, keep_dims=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = tf.mul(pi_samples[0], tf.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in xrange(K-1):
        s += tf.mul(pi_samples[k+1], tf.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -tf.log(s)


### Gaussian Mixture Model VAE Class
class GaussMMVAE(object):
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
                'kumar_a':init_mlp([hyperParams['hidden_d'], self.K-1]),                
                'kumar_b':init_mlp([hyperParams['hidden_d'], self.K-1])}


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
        
        for k in xrange(self.K):
            self.mu.append(mlp(h1, self.encoder_params['mu'][k]))
            self.sigma.append(tf.exp(mlp(h1, self.encoder_params['sigma'][k])))
            self.z.append(self.mu[-1] + tf.mul(self.sigma[-1], tf.random_normal(tf.shape(self.sigma[-1]))))
            x_recon_linear.append(mlp(self.z[-1], self.decoder_params))

        self.kumar_a = tf.exp(mlp(h1, self.encoder_params['kumar_a']))
        self.kumar_b = tf.exp(mlp(h1, self.encoder_params['kumar_b']))

        return x_recon_linear


    def compose_stick_segments(self, v):

        segments = []
        self.remaining_stick = [tf.ones((tf.shape(v)[0],1))]
        for i in xrange(self.K-1):
            curr_v = tf.slice(v, [0, i], [-1, -1])
            segments.append( tf.mul(curr_v, self.remaining_stick[-1]) )
            self.remaining_stick.append( tf.mul(1-curr_v, self.remaining_stick[-1]) )
        segments.append(self.remaining_stick[-1])

        return segments


    def get_ELBO(self):
        a_inv = tf.pow(self.kumar_a,-1)
        b_inv = tf.pow(self.kumar_b,-1)

        # compute Kumaraswamy means
        v_means = tf.mul(self.kumar_b, beta_fn(1.+a_inv, self.kumar_b))

        # compute Kumaraswamy samples
        uni_samples = tf.random_uniform(tf.shape(v_means), minval=1e-8, maxval=1-1e-8) 
        v_samples = tf.pow(1-tf.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)
        self.pi_means = self.compose_stick_segments(v_means)
        self.pi_samples = self.compose_stick_segments(v_samples)

    
        # compose elbo
        elbo = tf.mul(self.pi_means[0], compute_nll(self.X, self.x_recons_linear[0]) + gauss_cross_entropy(self.mu[0], self.sigma[0], self.prior['mu'][0], self.prior['sigma'][0]))
        for k in xrange(self.K-1):
            elbo += tf.mul(self.pi_means[k+1], compute_nll(self.X, self.x_recons_linear[k+1]) \
                               + gauss_cross_entropy(self.mu[k+1], self.sigma[k+1], self.prior['mu'][k+1], self.prior['sigma'][k+1]))
            elbo -= compute_kumar2beta_kld(tf.slice(self.kumar_a,[0,k],[-1,-1]), tf.slice(self.kumar_a,[0,k],[-1,-1]), \
                                               self.prior['dirichlet_alpha'], (self.K-k-1)*self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples, self.z, self.mu, self.sigma, self.K)

        return tf.reduce_mean(elbo)
