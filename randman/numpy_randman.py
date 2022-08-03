#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools



class NumpyRandman:
    """ Randman (numpy version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, use_bias=False, prec=1e-3, max_f_cutoff=1000):
        """ Initializes a randman object.
        
        Args
        ----
        embedding_dim : The embedding space dimension
        manifold_dim : The manifold dimension
        alpha : The power spectrum fall-off exponenent. Determines the smoothenss of the manifold (default 2)
        use_bias: If True, manifolds are placed at random offset coordinates within a [0,1] simplex.
        prec: The precision paramter to determine the maximum frequency cutoff (default 1e-3)
        """
        self.alpha = alpha
        self.use_bias = use_bias
        self.dim_embedding = embedding_dim
        self.dim_manifold = manifold_dim
        self.f_cutoff = int(np.min((np.ceil(np.power(prec,-1/self.alpha)),max_f_cutoff)))
        self.params_per_1d_fun = 3
        self.init_random()
           
    def init_random(self):
        self.params = np.random.uniform(low=0, high=1, size=(self.dim_embedding, self.dim_manifold, self.params_per_1d_fun, self.f_cutoff))
        if not self.use_bias:
            self.params[:,:,0,0] = 0
        
    def eval_random_function_1d(self, x, theta):
        tmp = np.zeros(len(x))
        s = 1.0/((np.arange(self.f_cutoff)+1)**self.alpha)
        for i in range(self.f_cutoff):
            tmp += theta[0,i]*s[i]*np.sin( 2*np.pi*(i*x*theta[1,i] + theta[2,i]) )
        return tmp

    def eval_random_function(self, x, params):
        tmp = np.ones(len(x))
        for d in range(self.dim_manifold):
            tmp *= self.eval_random_function_1d(x[:,d], params[d])
        return tmp
    
    def eval_manifold(self, x):
        dims = []
        for i in range(self.dim_embedding):
            dims.append(self.eval_random_function(x, self.params[i]))
        data = np.array( dims ).T
        return data
    
    def get_random_manifold_samples(self, nb_samples):
        x = np.random.rand(nb_samples,self.dim_manifold)
        y = self.eval_manifold(x)
        return x,y

