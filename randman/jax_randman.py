#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools
from functools import partial

import jax
import jax.numpy as jnp







class JaxRandman:
    """ Randman (jax version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, beta=0, prec=1e-3, max_f_cutoff=1000, use_bias=False, seed=0, dtype=jnp.float32):
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
        self.beta = beta
        self.use_bias = use_bias
        self.dim_embedding = embedding_dim
        self.dim_manifold = manifold_dim
        self.f_cutoff = int(np.min((np.ceil(np.power(prec,-1/self.alpha)),max_f_cutoff)))
        self.params_per_1d_fun = 3
        self.dtype=dtype
        
        self.key = jax.random.PRNGKey(seed)

        self.init_random()
        self.init_spect(self.alpha, self.beta)
           
    def init_random(self):
        self.params = jax.random.uniform(self.key, minval=0.0, maxval=1.0, shape=(self.dim_embedding, self.dim_manifold, self.params_per_1d_fun, self.f_cutoff))
        if not self.use_bias:
            self.params = self.params.at[:,:,0,0].set(0)

    def init_spect(self, alpha=2.0, res=0, ):
        """ Sets up power spectrum modulation 
        
        Args
        ----
        alpha : Power law decay exponent of power spectrum
        res : Peak value of power spectrum.
        """
        r = (jnp.arange(self.f_cutoff, dtype=self.dtype)+1)
        s = 1.0/(jnp.abs(r-res)**alpha + 1.0)
        self.spect = s


    def eval_freq(self, x, s, idx, theta):
        return theta[0] * s * jnp.sin( 2*jnp.pi*( idx * x * theta[1] + theta[2]) )
        
    def eval_random_function_1d(self, x, theta):
        s = self.spect
        idxs = jnp.arange(self.f_cutoff)
        tmp =  jax.vmap(self.eval_freq, in_axes=(None,0,0,1))(x, s, idxs, theta)
        return jnp.sum(tmp, axis=0)

    def eval_random_function(self, x, params):
        tmp = jax.vmap(self.eval_random_function_1d, in_axes=(1,0))(x, params)
        return jnp.prod(tmp, axis=0)
    
    @partial(jax.jit, static_argnums=0)
    def eval_manifold(self, x):
        x = jnp.array(x)
        return jax.vmap(self.eval_random_function, in_axes=(None, 0), out_axes=1)(x, self.params)
    
    def get_random_manifold_samples(self, nb_samples):

        x = jnp.empty((nb_samples, self.dim_manifold), dtype=self.dtype)
        y = jnp.empty((nb_samples, self.dim_embedding), dtype=self.dtype)
        q = 1000

        for i in range(nb_samples//q):
            key, subkey = jax.random.split(self.key)
            self.key = key
            sub_x = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(q, self.dim_manifold))
            sub_y = self.eval_manifold(sub_x)
            x = x.at[q*i:q*(i+1), :].set(sub_x)
            y = y.at[q*i:q*(i+1), :].set(sub_y)

        rem = nb_samples%q
        if rem>0:
            key, subkey = jax.random.split(self.key)
            self.key = key
            sub_x = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(rem, self.dim_manifold))
            sub_y = self.eval_manifold(sub_x)
            x = x.at[-rem:, :].set(sub_x)
            y = y.at[-rem:, :].set(sub_y)

        return x, y
