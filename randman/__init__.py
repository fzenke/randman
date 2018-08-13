#!/usr/bin/python3

import numpy as np
import pickle
import gzip
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import svm




def random_function_1d(x, alpha=2.0, bias_terms=False, freq_cutoff_=128):
    A = np.random.uniform(low=0, high=1, size=freq_cutoff_)
    B = np.random.uniform(low=0, high=1, size=freq_cutoff_)
    C = np.random.uniform(low=0, high=1, size=freq_cutoff_)
    if not bias_terms:
        A[0] = 0
    tmp = np.zeros(len(x))

    for i in range(freq_cutoff_):
        S = 1.0/(i+1)**alpha # pink noise spectrum
        tmp += A[i]*S*np.sin( 2*np.pi*(i*x*B[i] + C[i]) ) 
    return tmp


def random_function(x, alpha=2.0):
    tmp = np.ones(len(x))
    for d in range(x.shape[1]):
        tmp *= random_function_1d(x[:,d], alpha=alpha)
    return tmp


def random_manifold(x, dim_embedding_space, alpha=2.0):
    dims = [] 
    for i in range(dim_embedding_space):
        dims.append(random_function(x, alpha))
    data = np.array( dims ).T
    return data




def make_classification_dataset( n_classes=2, n_samples_per_class=1000, alpha=2.0, dim_manifold=1, dim_embedding_space=2 ):

    print("Generating random manifolds")
    manifold_points = np.random.rand(n_samples_per_class, dim_manifold)
    # manifold_points = np.tile(np.linspace(0, np.pi, n_samples_per_class),dim_manifold).reshape(n_samples_per_class, dim_manifold)
    

    data = []
    labels = []
    for i in range(n_classes):
        # Add bias input
        pure = random_manifold(manifold_points, dim_embedding_space, alpha)
        # tmp = (pure-np.min(pure))/(np.max(pure)-np.min(pure))
        data.append( pure )
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

    return ret


def write_to_zipfile(dataset, filename):
    print("Pickling...")
    fp = gzip.open("%s.pkl.gz"%filename,'wb')
    pickle.dump(dataset, fp)
    fp.close()


def load_from_zipfile(filename):
    print("Loading data set...")
    fp = gzip.open("%s.pkl.gz"%filename,'r')
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
    dataset = make_classification_dataset(2, dim_manifold=1, dim_embedding_space=3, alpha=1.0, n_samples_per_class=5000)

    print("Computing linear SCV error")
    acc = compute_linear_SVC_accuracy(dataset)
    print("Linear SVC training accuracy %f%%"%(100*acc))

    foo = plot_dataset(dataset, plot3d=True)
    plt.show()

    # filename = "randman"
    # print("Writing to zipped pickle...")
    # write_to_zipfile(dataset, filename)
    # print("Writing to ASCII files...")
    # write_gnuplot_file(dataset, filename)


if __name__ == '__main__':
    main()
