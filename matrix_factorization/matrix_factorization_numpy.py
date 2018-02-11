################################################################################################################
# An implementation of matrix factorization from Scratch
################################################################################################################

import numpy

"""
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

"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        ### FACTORIZATION ###
        for i in xrange(len(R)):         ## Range - Number of users
            for j in xrange(len(R[i])):     ## Range = number of ratings for each user
                if R[i][j] > 0:                ## If value of matrix entry is greater than 0
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j]) ## Calulate the error based on the existing matricies P & Q
                    # Update P,Q based on alpha,beta
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        ### REGULARIZATION TO AVOID OVE FITTING###
        for i in xrange(len(R)):         ## Range - Number of users
            for j in xrange(len(R[i])):     ## Range = number of ratings for each user
                if R[i][j] > 0:
                    # adding a parameter \beta and modify the squared error as follows:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T





###################
# EXAMPLE
###################
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K) #Intialise a random matrix for users
    Q = numpy.random.rand(M,K) #Intialise a random matrix for items

    nP, nQ = matrix_factorization(R, P, Q, K)

    R=X
    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K) #Intialise a random matrix for users
    Q = numpy.random.rand(M,K) #Intialise a random matrix for items

    nP, nQ = matrix_factorization(R, P, Q, K)

    nQ


################################################################################################################
# sklearn Non-Negative Matrix Factorisation
################################################################################################################
"""
Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X.
This factorization can be used for example for dimensionality reduction, source separation or topic extraction.


# Objective function
0.5 * ||X - WH||_Fro^2
+ alpha * l1_ratio * ||vec(W)||_1
+ alpha * l1_ratio * ||vec(H)||_1
+ 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
+ 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

# Where:
||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
"""


import numpy as np
from sklearn.decomposition import NMF

X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

model = NMF(n_components=6, init='random', random_state=0)
model.fit(X)
print model.components_

#reconstruction_err_
#Frobenius norm of the matrix difference between the training data and the reconstructed data from the fit produced by the model. || X - WH ||_2
print model.reconstruction_err_

# Actual number of iterations.
print model.n_iter_






#
