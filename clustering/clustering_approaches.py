#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Matrix Factorisation using Numpy and scikit
    An implementation of matrix factorization from Scratch

    -----------
    Parameters:
    -----------
    R     : a matrix to be factorized, dimension N x M (Users N, Items M)
    P     : an initial matrix of dimension N x K       (User feature matrix)
    Q     : an initial matrix of dimension M x K       (Item feature matrix)
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter

    Returns:
    -----------
    the final matrices P and Q



# cluster and plot
from sklearn.cluster import KMeans
km = KMeans(25, init='k-means++') # initialize
km.fit(cluster_data)
c = km.predict(cluster_data)
dest_agg3['cluster']=c

#plot
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(dest_agg3[x_name], dest_agg3[y_name], c=c)
