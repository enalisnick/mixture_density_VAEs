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
        #h.append( tf.nn.relu( tf.matmul(h[-1], w) + b ) )
        h.append( tf.nn.tanh( tf.matmul(h[-1], w) + b ) ) 
    return tf.matmul(h[-1], params['w'][-1]) + params['b'][-1]


def compute_nll(x, x_recon_linear):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_recon_linear, x), reduction_indices=1, keep_dims=True)


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


def log_beta_pdf(v, alpha, beta):
    return tf.reduce_sum((alpha-1)*tf.log(v) + (beta-1)*tf.log(1-v) - tf.log(beta_fn(alpha,beta)), reduction_indices=1, keep_dims=True)

def log_kumar_pdf(v, a, b):
    return tf.reduce_sum(tf.mul(a-1, tf.log(v)) + tf.mul(b-1, tf.log(1-tf.pow(v,a))) + tf.log(a) + tf.log(b), reduction_indices=1, keep_dims=True)


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

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()

        #self.batch_log_margLL = self.get_log_margLL(hyperParams['batchSize'])


    def init_encoder(self, hyperParams):
        return {'base':init_mlp([hyperParams['input_d'], hyperParams['hidden_d'], hyperParams['hidden_d']]), 
                'mu':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'sigma':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'kumar_a':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8),                
                'kumar_b':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8)}


    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def f_prop(self):

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
            curr_v = tf.expand_dims(v[:,i],1)
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
        elbo = tf.mul(self.pi_means[0], -compute_nll(self.X, self.x_recons_linear[0]) + gauss_cross_entropy(self.mu[0], self.sigma[0], self.prior['mu'][0], self.prior['sigma'][0]))
        for k in xrange(self.K-1):
            elbo += tf.mul(self.pi_means[k+1], -compute_nll(self.X, self.x_recons_linear[k+1]) \
                               + gauss_cross_entropy(self.mu[k+1], self.sigma[k+1], self.prior['mu'][k+1], self.prior['sigma'][k+1]))
            elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples, self.z, self.mu, self.sigma, self.K)

        return tf.reduce_mean(elbo)


    def get_log_margLL(self, batchSize):
        a_inv = tf.pow(self.kumar_a,-1)
        b_inv = tf.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples                                                                
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K-1), minval=1e-8, maxval=1-1e-8)
        v_samples = tf.pow(1-tf.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)                                                     
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K), minval=1e-8, maxval=1-1e-8)
        gumbel_samples = -tf.log(-tf.log(uni_samples))
        component_samples = tf.to_int32(tf.argmax(tf.log(tf.concat(1, self.pi_samples)) + gumbel_samples, 1))

        # calc likelihood term for chosen components
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = tf.concat(1, all_ll)

        component_samples = tf.concat(1, [tf.expand_dims(tf.range(0,batchSize),1), tf.expand_dims(component_samples,1)])
        ll = tf.gather_nd(all_ll, component_samples)
        ll = tf.expand_dims(ll,1)

        # calc prior terms
        all_log_gauss_priors = [] 
        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = tf.concat(1, all_log_gauss_priors)
        log_gauss_prior = tf.gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = tf.expand_dims(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), self.prior['dirichlet_alpha'], (self.K-1)*self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), self.prior['dirichlet_alpha'], (self.K-2-k)*self.prior['dirichlet_alpha'])

        # calc post term
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)
        
        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.mu[k], self.sigma[k]))
        all_log_gauss_posts = tf.concat(1, all_log_gauss_posts)
        log_gauss_post = tf.gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tf.expand_dims(log_gauss_post,1)

        return ll + log_beta_prior + log_gauss_prior - log_kumar_post - log_gauss_post

    def get_samples(self, nImages):
        samples_from_each_component = []
        for k in xrange(self.K): 
            z = self.prior['mu'][k] + tf.mul(self.prior['sigma'][k], tf.random_normal((nImages, tf.shape(self.decoder_params['w'][0])[0]))) 
            samples_from_each_component.append( tf.sigmoid(mlp(z, self.decoder_params)) )
        return samples_from_each_component


class DPVAE(GaussMMVAE):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()



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
        elbo = tf.mul(self.pi_means[0], -compute_nll(self.X, self.x_recons_linear[0]) + gauss_cross_entropy(self.mu[0], self.sigma[0], self.prior['mu'][0], self.prior['sigma'][0]))
        for k in xrange(self.K-1):
            elbo += tf.mul(self.pi_means[k+1], -compute_nll(self.X, self.x_recons_linear[k+1]) \
                               + gauss_cross_entropy(self.mu[k+1], self.sigma[k+1], self.prior['mu'][k+1], self.prior['sigma'][k+1]))
            elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a[:,k],1), tf.expand_dims(self.kumar_b[:,k],1), 1., self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples, self.z, self.mu, self.sigma, self.K)

        return tf.reduce_mean(elbo)


    def get_log_margLL(self, batchSize):
        a_inv = tf.pow(self.kumar_a,-1)
        b_inv = tf.pow(self.kumar_b,-1)

        # compute Kumaraswamy samples                                                                                                                                                           
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K-1), minval=1e-8, maxval=1-1e-8)
        v_samples = tf.pow(1-tf.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)                                                                                                                                  
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index                                                                                                                                                              
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K), minval=1e-8, maxval=1-1e-8)
        gumbel_samples = -tf.log(-tf.log(uni_samples))
        component_samples = tf.to_int32(tf.argmax(tf.log(tf.concat(1, self.pi_samples)) + gumbel_samples, 1))

        # calc likelihood term for chosen components                                                                                                                                            
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = tf.concat(1, all_ll)

        component_samples = tf.concat(1, [tf.expand_dims(tf.range(0,batchSize),1), tf.expand_dims(component_samples,1)])
        ll = tf.gather_nd(all_ll, component_samples)
        ll = tf.expand_dims(ll,1)

        # calc prior terms                                                                                                                                                                      
        all_log_gauss_priors = []
        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = tf.concat(1, all_log_gauss_priors)
        log_gauss_prior = tf.gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = tf.expand_dims(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), 1., self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), 1., self.prior['dirichlet_alpha'])

        # calc post term                                                            
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)

        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.mu[k], self.sigma[k]))
        all_log_gauss_posts = tf.concat(1, all_log_gauss_posts)
        log_gauss_post = tf.gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tf.expand_dims(log_gauss_post,1)

        return ll + log_beta_prior + log_gauss_prior - log_kumar_post - log_gauss_post


### *DEEP* Latent Gaussian MM
class DLGMM(GaussMMVAE):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.prior = hyperParams['prior']
        self.K = hyperParams['K']

        self.encoder_params1 = self.init_encoder(hyperParams)
        self.encoder_params2 = self.init_encoder(hyperParams)
        self.decoder_params1 = self.init_decoder(hyperParams)
        self.decoder_params1 = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop()

        self.elbo_obj = self.get_ELBO()

        #self.batch_log_margLL = self.get_log_margLL(hyperParams['batchSize'])                                                                                             


    def init_encoder(self, hyperParams):
        return {'base':init_mlp([hyperParams['input_d'], hyperParams['hidden_d']]),
                'mu':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'sigma':[init_mlp([hyperParams['hidden_d'], hyperParams['latent_d']]) for k in xrange(self.K)],
                'kumar_a':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8),
                'kumar_b':init_mlp([hyperParams['hidden_d'], self.K-1], 1e-8)}


    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['latent_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def f_prop(self):
        # init variational params                                                                                                                                   
        self.mu1 = []
        self.mu2 =[]
        self.sigma1 = []
        self.sigma2 = []
        self.z1 = []
        self.z2 = []
        self.kumar_a1 = []
        self.kumar_b1 = []
        x_recon_linear = []

        h1 = mlp(self.X, self.encoder_params1['base'])

        # compute z1's params
        for k in xrange(self.K):
            #self.mu1.append(mlp(h1, self.encoder_params1['mu'][k]))
            self.sigma1.append(tf.exp(mlp(h1, self.encoder_params1['sigma'][k])))

        h2 = mlp(h1, self.encoder_params2['base'])
            
        # compute z2's param                                                                                     
        for k in xrange(self.K):
            self.mu2.append(mlp(h2, self.encoder_params2['mu'][k]))
            self.sigma2.append(tf.exp(mlp(h2, self.encoder_params2['sigma'][k])))
            self.z2.append(self.mu2[-1] + tf.mul(self.sigma2[-1], tf.random_normal(tf.shape(self.sigma2[-1]))))
        self.kumar_a2 = tf.exp(mlp(h2, self.encoder_params2['kumar_a']))
        self.kumar_b2 = tf.exp(mlp(h2, self.encoder_params2['kumar_b']))

        h3 = []
        for k in xrange(self.K):
            h3.append(mlp(self.z2[k], self.decoder_params2))

        # compute z1's, finally.  KxK of them
        for k in xrange(self.K):
            self.z1.append([])
            self.mu1.append([])
            self.kumar_a1.append(tf.exp(mlp(h3[k], self.encoder_params1['kumar_a'])))
            self.kumar_b1.append(tf.exp(mlp(h3[k], self.encoder_params1['kumar_a'])))
            for j in xrange(self.K):
                self.mu1[-1].append(mlp(self.h3[k], self.encoder_params1['mu'][j]))
                self.z1[-1].append(self.mu1[-1][-1] + tf.mul(self.sigma1[k], tf.random_normal(tf.shape(self.sigma1[k]))))
        
        # compute KxK reconstructions
        for k in xrange(self.K):
            x_recon_linear.append([])
            for j in xrange(self.K):
                x_recon_linear[-1].append(mlp(self.z[k][j], self.decoder_params1))

        return x_recon_linear


    def get_ELBO(self):
        a1_inv = tf.pow(self.kumar_a1,-1)
        a2_inv = tf.pow(self.kumar_a2,-1)
        b1_inv = tf.pow(self.kumar_b1,-1)
        b2_inv = tf.pow(self.kumar_b2,-1)

        # compute Kumaraswamy means                                                                                                                       
        v_means1 = tf.mul(self.kumar_b1, beta_fn(1.+a_inv1, self.kumar_b1))
        v_means2 = tf.mul(self.kumar_b2, beta_fn(1.+a_inv2, self.kumar_b2))

        # compute Kumaraswamy samples                                                                                                                  
        uni_samples1 = tf.random_uniform(tf.shape(v_means1), minval=1e-8, maxval=1-1e-8)
        uni_samples2 = tf.random_uniform(tf.shape(v_means2), minval=1e-8, maxval=1-1e-8)
        v_samples1 = tf.pow(1-tf.pow(uni_samples1, b1_inv), a1_inv)
        v_samples2 = tf.pow(1-tf.pow(uni_samples2, b2_inv), a2_inv)

        # compose into stick segments using pi = v \prod (1-v)                                                                                         
        self.pi_means1 = self.compose_stick_segments(v_means1)
        self.pi_samples1 = self.compose_stick_segments(v_samples1)
        self.pi_means2 = self.compose_stick_segments(v_means2)
        self.pi_samples2 = self.compose_stick_segments(v_samples2)

        # compose elbo                                                                                
        elbo = tf.zeros(tf.shape(self.pi_means1[0]))
        for k in xrange(self.K):
            elbo += tf.mul(self.pi_means2[k], gauss_cross_entropy(self.mu2[k], self.sigma2[k], self.prior['mu'][k], self.prior['sigma'][k]))
            for j in xrange(self.K):
                elbo += tf.mul(tf.mul(self.pi_means1[k], self.pi_means2[j]), -compute_nll(self.X, self.x_recons_linear[k][j]))
                elbo += tf.mul(tf.mul(self.pi_means1[k], self.pi_means2[j]), log_normal_pdf(self.z1[k][j], self.mu1[k],[j], self.sigma1[k]))
        
        for k in xrange(self.K-1):
            elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a1[:,k],1), tf.expand_dims(self.kumar_b1[:,k],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])
            elbo -= compute_kumar2beta_kld(tf.expand_dims(self.kumar_a2[:,k],1), tf.expand_dims(self.kumar_b2[:,k],1), \
                                               self.prior['dirichlet_alpha'], (self.K-1-k)*self.prior['dirichlet_alpha'])

        elbo += mcMixtureEntropy(self.pi_samples2, self.z2, self.mu2, self.sigma2, self.K)
        for k in xrange(self.K):
            elbo += tf.mul(self.pi_means2[k], mcMixtureEntropy(self.pi_samples1, self.z1[k], self.mu1[k], self.sigma, self.K))

        return tf.reduce_mean(elbo)


    def get_log_margLL(self, batchSize):
        a_inv1 = tf.pow(self.kumar_a1,-1)
        b_inv1 = tf.pow(self.kumar_b1,-1)
        a_inv2 = tf.pow(self.kumar_a2,-1)
        b_inv2 = tf.pow(self.kumar_b2,-1)

        # compute Kumaraswamy samples                                                                                                                                
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K-1), minval=1e-8, maxval=1-1e-8)
        v_samples = tf.pow(1-tf.pow(uni_samples, b_inv), a_inv)

        # compose into stick segments using pi = v \prod (1-v)                                                                                             
        self.pi_samples = self.compose_stick_segments(v_samples)

        # sample a component index                                                                                                                                
        uni_samples = tf.random_uniform((tf.shape(a_inv)[0], self.K), minval=1e-8, maxval=1-1e-8)
        gumbel_samples = -tf.log(-tf.log(uni_samples))
        component_samples = tf.to_int32(tf.argmax(tf.log(tf.concat(1, self.pi_samples)) + gumbel_samples, 1))

        # calc likelihood term for chosen components                                                                                                                    
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = tf.concat(1, all_ll)

        component_samples = tf.concat(1, [tf.expand_dims(tf.range(0,batchSize),1), tf.expand_dims(component_samples,1)])
        ll = tf.gather_nd(all_ll, component_samples)
        ll = tf.expand_dims(ll,1)

        # calc prior terms                                                                                                                                  
        all_log_gauss_priors = []
        for k in xrange(self.K):
            all_log_gauss_priors.append(log_normal_pdf(self.z[k], self.prior['mu'][k], self.prior['sigma'][k]))
        all_log_gauss_priors = tf.concat(1, all_log_gauss_priors)
        log_gauss_prior = tf.gather_nd(all_log_gauss_priors, component_samples)
        log_gauss_prior = tf.expand_dims(log_gauss_prior,1)

        log_beta_prior = log_beta_pdf(tf.expand_dims(v_samples[:,0],1), 1., self.prior['dirichlet_alpha'])
        for k in xrange(self.K-2):
            log_beta_prior += log_beta_pdf(tf.expand_dims(v_samples[:,k+1],1), 1., self.prior['dirichlet_alpha'])

        # calc post term                                                                                                                  
        log_kumar_post = log_kumar_pdf(v_samples, self.kumar_a, self.kumar_b)

        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.mu[k], self.sigma[k]))
        all_log_gauss_posts = tf.concat(1, all_log_gauss_posts)
        log_gauss_post = tf.gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tf.expand_dims(log_gauss_post,1)

        return ll + log_beta_prior + log_gauss_prior - log_kumar_post - log_gauss_post
