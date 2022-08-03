#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools

# from randman.numpy_randman import NumpyRandman
# from randman.jax_randman import JaxRandman
from randman.torch_randman import TorchRandman



# Defines class alias for the default backend
Randman = TorchRandman



def make_classification_dataset( nb_classes=2, nb_samples_per_class=1000, alpha=2.0, dim_manifold=1, dim_embedding_space=2 ):

    print("Generating random manifolds")

    data = []
    labels = []
    for i in range(nb_classes):
        randman = Randman(dim_embedding_space, dim_manifold, alpha=alpha)
        _,tmp = randman.get_random_manifold_samples(nb_samples_per_class)
        data.append( tmp )
        labels.append( i*np.ones(nb_samples_per_class) )

    
    print("Shuffling dataset")
    X = np.vstack(data)
    Y = np.hstack(labels)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

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
    from sklearn import svm

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
    dataset = make_classification_dataset(2, dim_manifold=2, dim_embedding_space=3, alpha=2.0, nb_samples_per_class=1000)

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
