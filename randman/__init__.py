#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools

import torch


from sklearn import svm


class NumpyRandman:
    """ Randman (numpy version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, use_bias=False, prec=1e-3):
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
        self.f_cutoff = int(np.ceil(np.power(prec,-1/self.alpha)))
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

    
    
    
class TorchRandman:
    """ Randman (torch version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled. """
    
    def __init__(self, embedding_dim, manifold_dim, alpha=2, prec=1e-3, use_bias=False, dtype=torch.float32, device=None):
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
        self.f_cutoff = int(np.ceil(np.power(prec,-1/self.alpha)))
        self.params_per_1d_fun = 3
        self.dtype=dtype
        if device is None:
            self.device=torch.device("cpu")
        else:
            self.device=device
        self.init_random()
           
    def init_random(self):
        self.params = torch.rand(self.dim_embedding, self.dim_manifold, self.params_per_1d_fun, self.f_cutoff, dtype=self.dtype, device=self.device)
        if not self.use_bias:
            self.params[:,:,0,0] = 0
        
    def eval_random_function_1d(self, x, theta):       
        tmp = torch.zeros(len(x),dtype=self.dtype,device=self.device)
        s = 1.0/((torch.arange(self.f_cutoff,dtype=self.dtype,device=self.device)+1)**self.alpha)
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


# Defines class alias for the default backend
Randman = TorchRandman



def make_classification_dataset( n_classes=2, n_samples_per_class=1000, alpha=2.0, dim_manifold=1, dim_embedding_space=2 ):

    print("Generating random manifolds")

    data = []
    labels = []
    for i in range(n_classes):
        randman = Randman(dim_embedding_space, dim_manifold, alpha=alpha)
        _,tmp = randman.get_random_manifold_samples(n_samples_per_class)
        data.append( tmp )
        labels.append( i*np.ones(n_samples_per_class) )

    
    print("Shuffling dataset")
    X = np.vstack(data)
    Y = np.hstack(labels)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    print("Standardizing the data")
    # X -= X.mean(0)
    # X /= (X.std(0)+1e-9)
    X -= X.min(0)
    X /= (X.max(0))

    # print("Add fixed input")
    # X = np.hstack((X,np.ones((len(X),1))))
    # print X.shape

    dataset = (X,Y)

    return dataset


def plot_dataset(dataset, plot3d=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    data, labels = dataset

    print("Plotting manifolds")
    fig = plt.figure()

    if plot3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # Plot manifolds
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y", "k"])
    n_plot = np.min((5000, len(data)))-1
    n_classes = len(np.unique(labels))

    for i in range(n_classes):
        idx = (labels==i)
        if len(idx) > n_plot:
            idx[n_plot:] = False
        if plot3d:    
            ret = ax.scatter( data[idx,0], data[idx,1], data[idx,2], s=0.5, color=next(colors) )
        else:
            ret = ax.scatter( data[idx,0], data[idx,1], s=0.5, color=next(colors) )

    plt.show()


def write_to_zipfile(dataset, filename):
    print("Pickling...")
    fp = gzip.open("%s"%filename,'wb')
    pickle.dump(dataset, fp)
    fp.close()


def load_from_zipfile(filename):
    print("Loading data set...")
    fp = gzip.open("%s"%filename,'r')
    dataset = pickle.load(fp)
    fp.close()
    return dataset


def write_gnuplot_file(dataset, filename):
    data, labels = dataset
    fp = open("%s.dat"%filename,'w')
    for i,d in enumerate(data):
        for val in d:
            fp.write("%f "%val)
        fp.write("%i\n"%labels[i])
    fp.close()


def compute_linear_SVC_accuracy(dataset):
    X,Y = dataset

    # Splitting into training set and held out data
    n_data  = len(X)
    x_train = X[:n_data//4*3]
    x_test  = X[n_data//4*3:]
    y_train = Y[:n_data//4*3]
    y_test  = Y[n_data//4*3:]
    train = (x_train, y_train) 
    test  = (x_test, y_test)

    # Fit SVC and evaluate on test set
    lin_svc = svm.LinearSVC(C=1.0).fit(train[0], train[1])
    pred = lin_svc.predict( test[0] )
    acc = np.mean(pred==test[1])
    return acc



def main():
    dataset = make_classification_dataset(2, dim_manifold=2, dim_embedding_space=3, alpha=2.0, n_samples_per_class=1000)

    print("Computing linear SCV error")
    acc = compute_linear_SVC_accuracy(dataset)
    print("Linear SVC training accuracy %f%%"%(100*acc))

    foo = plot_dataset(dataset, plot3d=True)

    # filename = "randman"
    # print("Writing to zipped pickle...")
    # write_to_zipfile(dataset, filename)
    # print("Writing to ASCII files...")
    # write_gnuplot_file(dataset, filename)


if __name__ == '__main__':
    main()
