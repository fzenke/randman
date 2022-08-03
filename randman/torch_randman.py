#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools

import torch





class TorchRandman:
    """ Randman (torch version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, beta=0, prec=1e-3, max_f_cutoff=1000, use_bias=False, seed=None, dtype=torch.float32, device=None):
        """ Initializes a randman object.
        
        Args
        ----
        embedding_dim : The embedding space dimension
        manifold_dim : The manifold dimension
        alpha : The power spectrum fall-off exponenent. Determines the smoothenss of the manifold (default 2)
        use_bias: If True, manifolds are placed at random offset coordinates within a [0,1] simplex.
        seed: This seed is used to init the *global* torch.random random number generator. 
        prec: The precision paramter to determine the maximum frequency cutoff (default 1e-3)
        """
        self.alpha = alpha
        self.beta = beta
        self.use_bias = use_bias
        self.dim_embedding = embedding_dim
        self.dim_manifold = manifold_dim
        self.f_cutoff = int(np.min((np.ceil(np.power(prec,-1/self.alpha)),max_f_cutoff)))
        self.params_per_1d_fun = 3
        self.dtype=dtype
        
        if device is None:
            self.device=torch.device("cpu")
        else:
            self.device=device
        
        if seed is not None:
            torch.random.manual_seed(seed)

        self.init_random()
        self.init_spect(self.alpha, self.beta)
           
    def init_random(self):
        self.params = torch.rand(self.dim_embedding, self.dim_manifold, self.params_per_1d_fun, self.f_cutoff, dtype=self.dtype, device=self.device)
        if not self.use_bias:
            self.params[:,:,0,0] = 0

    def init_spect(self, alpha=2.0, res=0, ):
        """ Sets up power spectrum modulation 
        
        Args
        ----
        alpha : Power law decay exponent of power spectrum
        res : Peak value of power spectrum.
        """
        r = (torch.arange(self.f_cutoff,dtype=self.dtype,device=self.device)+1)
        s = 1.0/(torch.abs(r-res)**alpha + 1.0)
        self.spect = s
        
    def eval_random_function_1d(self, x, theta):       
        tmp = torch.zeros(len(x),dtype=self.dtype,device=self.device)
        s = self.spect
        for i in range(self.f_cutoff):
            tmp += theta[0,i]*s[i]*torch.sin( 2*np.pi*(i*x*theta[1,i] + theta[2,i]) )
        return tmp

    def eval_random_function(self, x, params):
        tmp = torch.ones(len(x),dtype=self.dtype,device=self.device)
        for d in range(self.dim_manifold):
            tmp *= self.eval_random_function_1d(x[:,d], params[d])
        return tmp
    
    def eval_manifold(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x,dtype=self.dtype,device=self.device)
        tmp = torch.zeros((x.shape[0],self.dim_embedding),dtype=self.dtype,device=self.device)
        for i in range(self.dim_embedding):
            tmp[:,i] = self.eval_random_function(x, self.params[i])
        return tmp
    
    def get_random_manifold_samples(self, nb_samples):
        x = torch.rand(nb_samples,self.dim_manifold,dtype=self.dtype,device=self.device)
        y = self.eval_manifold(x)
        return x,y

