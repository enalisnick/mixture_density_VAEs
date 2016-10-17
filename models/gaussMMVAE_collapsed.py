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
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_recon_linear, x), 1)


def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    d = mu_post - mu_prior
    return tf.reduce_sum((d*d)/(2.*sigma_prior*sigma_prior) + sigma_post*sigma_post, 1)


def compute_kumar2beta_kld(a, b, alpha, beta):
    # compute taylor expansion for E[log (1-v)] term                                                                               
    kl = 1./(1+a*b) * tf.exp(tf.lbeta(tf.concat(1,[1./a, b])))
    kl += 1./(2+a*b) * tf.exp(tf.lbeta(tf.concat(1,[2./a, b])))
    kl += 1./(3+a*b) * tf.exp(tf.lbeta(tf.concat(1,[3./a, b])))
    kl += 1./(4+a*b) * tf.exp(tf.lbeta(tf.concat(1,[4./a, b])))   
    kl += 1./(5+a*b) * tf.exp(tf.lbeta(tf.concat(1,[5./a, b])))
    kl += 1./(6+a*b) * tf.exp(tf.lbeta(tf.concat(1,[6./a, b])))
    kl += 1./(7+a*b) * tf.exp(tf.lbeta(tf.concat(1,[7./a, b])))
    kl += 1./(8+a*b) * tf.exp(tf.lbeta(tf.concat(1,[8./a, b])))
    kl += 1./(9+a*b) * tf.exp(tf.lbeta(tf.concat(1,[9./a, b])))
    kl += 1./(10+a*b) * tf.exp(tf.lbeta(tf.concat(1,[10./a, b])))
    kl *= (beta-1)*b

    # use another taylor approx for Digamma function                                                                                     
    kl += (a-alpha)/a * (-0.57721 - tf.digamma(b) - 1./b)
    # add normalization constants                                                                                                                         
    kl += tf.log(a*b) + tf.lbeta(tf.concat(1,[alpha*tf.ones((1,1)), beta*tf.ones((1,1))]))

    # final term                                                                                                  
    kl += -(b-1)/b

    return kl

def normal_pdf(x, mu, sigma):
    d = mu - x
    s2 = sigma*sigma
    return tf.reduce_prod(tf.exp(d*d/(2.*s2)) * 1./tf.sqrt(2*s2*3.14), 1)


def mcMixtureEntropy(pi_samples, z, mu, sigma):
    s = 0.
    for k in xrange(len(mu)):
        s += pi_samples[k] * normal_pdf(z[k], mu[k], sigma[k])
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
            self.z.append(self.mu[-1] + self.sigma[-1] * tf.random_normal(tf.shape(self.sigma[-1])))
            x_recon_linear.append(mlp(self.z[-1], self.decoder_params))

        self.kumar_a = tf.exp(mlp(h1, self.encoder_params['kumar_a']))
        self.kumar_b = tf.exp(mlp(h1, self.encoder_params['kumar_b']))

        return x_recon_linear


    def compose_stick_segments(self, v):
        #initializer = ( tf.ones((tf.shape(v,1),1)), tf.zeros((tf.shape(v,1),1)) )
        #return tf.scan( lambda a, _:  )

        segments = []
        remaining_stick = [tf.ones((tf.shape(v)[0],1))]
        for i in xrange(self.K-1):
            curr_v = tf.slice(v, [0, i], [-1, -1])
            segments.append( curr_v * remaining_stick )
            remaining_stick.append( (1-curr_v) * remaining_stick[-1] )
        segments.append(remaining_stick[-1])

        return segments


    def get_ELBO(self):
        
        # compute Kumaraswamy means
        v_means = tf.exp( tf.log(self.kumar_b) + tf.lbeta(tf.concat(1, [1+1./self.kumar_a, self.kumar_b])))

        # compute Kumaraswamy samples
        uni_samples = tf.random_uniform(tf.shape(v_means), minval=1e-8, maxval=1-1e-8) 
        v_samples = tf.pow(1-tf.pow(uni_samples, 1./self.kumar_b), 1./self.kumar_a)

        # compose into stick segments using pi = v \prod (1-v)
        pi_means = self.compose_stick_segments(v_means)
        pi_samples = self.compose_stick_segments(v_samples)

        # compose elbo
        elbo = tf.zeros([tf.size(pi_samples)[0],])
        tf.Print(elbo)
        for k in xrange(self.K):
            elbo += pi_means[k] * ( compute_nll(self.X, self.x_recons_linear[k]) + gauss_cross_entropy(self.mu[k], self.sigma[k], self.prior['mu'][k], self.prior['sigma'][k]) )
            if k < self.K-1:
                elbo -= compute_kumar2beta_kld(tf.slice(self.kumar_a,[0,k],[-1,-1]), tf.slice(self.kumar_a,[0,k],[-1,-1]), \
                                                   self.prior['dirichlet_alpha'], (self.K-k-1)*self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(pi_samples, self.z, self.mu, self.sigma)
        
        return tf.reduce_mean(elbo)
