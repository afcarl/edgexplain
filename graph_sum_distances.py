'''
Created on 4 Feb 2016
I want to check if sigma wij * ||Yi - Yj||2  (weighted sum of distances between nodes and neighbours in a graph) can be vectorized
@author: af
'''
import numpy as np

def weighted_sum_of_distances_loop(A, W):
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == W.shape[0]
    num_nodes = A.shape[0]
    total_error = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            aij = A[i, j]
            error_square = np.sum((W[i] - W[j])**2)
            total_error += aij * error_square
    print 'total_error', total_error
    
def weighted_sum_of_distances_vectorized(A, W):
    D = np.diag(A.sum(axis=0))
    D_bar = np.diag(A.sum(axis=1))
    L = D + D_bar - A - A.transpose()
    total_error = 0.0
    for l in range(W.shape[1]):
        Wl = W[:, l].reshape(W.shape[0], 1)
        Wl_t = W[:, l].reshape(1, W.shape[0])
        total_error += np.dot(np.dot(Wl_t, L) , Wl)
    print 'total_error', total_error    
    

A=np.array([[0, 15, 0, 7, 10, 0], [15, 0, 9, 11, 0, 9], [0, 9, 0, 0, 12, 7], [7, 11, 0, 0, 8, 14], [10, 0, 12, 8, 0, 8], [0, 9, 7, 14, 8, 0]])
W = np.random.rand(A.shape[0], 10)
weighted_sum_of_distances_loop(A, W)
weighted_sum_of_distances_vectorized(A, W)