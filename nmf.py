'''
Created on 25 Jan 2016

@author: code copied from http://un-mindlab.blogspot.com.au/
multiplicative update rules from Seung and Lee
'''
import theano
import theano.tensor as T
import numpy as np
import time
import sys

def NMFN(X,r,iterations,H=None,W=None):
    '''
    numpy implementation
    '''
    rng = np.random
    n = np.size(X,0)
    m = np.size(X,1)
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)

    for i in range(0,iterations):
        print 'iter', i, np.linalg.norm(X-np.dot(W,H))
        H = H*(np.dot(W.T,X)/np.dot(np.dot(W.T,W),H))
        W = W*(np.dot(X,H.T)/np.dot(np.dot(W,H),H.T))

    return W,H

def NMF(X,r,iterations,H=None,W=None):
    rng = np.random
    n = np.size(X,0)
    m = np.size(X,1)
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tH = theano.shared(H,name="H")
    tW = theano.shared(W,name="W")
    tE = T.sqrt(((tX-T.dot(tW,tH))**2).sum())

    trainH = theano.function(
            inputs=[],
            outputs=[tE],
            updates={tH:tH*((T.dot(tW.T,tX))/(T.dot(T.dot(tW.T,tW),tH)))},
            name="trainH")
    trainW = theano.function(
            inputs=[],
            outputs=[tE],
            updates={tW:tW*((T.dot(tX,tH.T))/(T.dot(tW,T.dot(tH,tH.T))))},
            name="trainW")

    for i in range(0,iterations):
        print 'iter: ', i, np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(),tH.get_value()

def NMF_regularized(X,r,iterations,H=None,W=None, learning_rate=0.1):
    rng = np.random
    n = np.size(X,0)
    m = np.size(X,1)
    iterations *= 4
    lambda1 = 0.0
    lambda2 = 0.0
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tH = theano.shared(H,name="H")
    tW = theano.shared(W,name="W")
    tRegularizer = lambda1 * T.abs_(tW).sum() + lambda2 * (tH**2).sum() 
    tEmbedding = T.sqrt(((tX-T.dot(tW,tH))**2).sum())
    tCost = tEmbedding + tRegularizer
    tGamma = T.scalar(name="learning_rate")
    tgrad_H, tgrad_W = T.grad(cost=tCost, wrt=[tH, tW]) 

    trainH = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tH:tH - tGamma * tgrad_H},
            name="trainH")
    trainW = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tW:tW - tGamma * tgrad_W},
            name="trainW")

    for i in range(0,iterations):
        tCostH = trainH(np.asarray(learning_rate,dtype=theano.config.floatX));
        tCostW = trainW(np.asarray(learning_rate,dtype=theano.config.floatX));
        print 'iter ' + str(i) + ':', np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        

    return tW.get_value(),tH.get_value()

def NMF_edgexplain(X, r, iterations, A, H=None, W=None, learning_rate=0.1, alpha=10, c=0):
    '''
    alpha and c are edgeexplain variables
    H and W are document-topic and topic-word matrices which should be inferred from input matrix X
    r is the embedding dimensionality (e.g. number of topics)
    A is the adjacancy matrix (document-document relations)
    '''
    rng = np.random
    n = X.shape[0]
    m = X.shape[1]
    #because we have more terms in the cost function, the model converges slower and needs more iterations.
    iterations *= 6
    #coefficients
    lambda1 = 0.001
    lambda2 = 0.001
    #note if lambda3 is high the model doesn't converge
    lambda3 = 0.0001
    if(H is None):
        H = rng.random((r,m)).astype(theano.config.floatX)
    if(W is None):
        W = rng.random((n,r)).astype(theano.config.floatX)
    
    use_regularized_ridge_regression_for_h = False
    I = np.identity(r).astype(theano.config.floatX)
    tI = theano.shared(I, name="I")
    tX = theano.shared(X.astype(theano.config.floatX),name="X")
    tH = theano.shared(H,name="H")
    tW = theano.shared(W,name="W")
    tA = theano.shared(A, name="A")
    
    tL1 = T.sum(abs(tW))
    smooth_out_L1 = False
    if smooth_out_L1:
        epsilon = 0.1
        tL1 = T.sqrt(tW ** 2 + epsilon).sum()
    tL2 = T.sum(tH ** 2)
    '''
    Note: we don't need the sqrt in the cost function maximizing x^2 is similar to maximizing sqrt(x^2)... 
    but the function becomes very big and becomes inf if we don't do the sqrt.
    One possible solution is to divide it by a very big number to avoid inf.
    '''
    tEmbedding = T.sqrt(((tX-T.dot(tW,tH))**2).sum())
    #tEmbedding = ((tX-T.dot(tW,tH))**2).sum()
    tRegularizer = lambda1 * tL1 + lambda2 * tL2 
    #tEdgexplain = lambda3 * (T.log(1.0 / (1 + T.exp(-(c + alpha * tA * T.dot(tW, tW.transpose())))))).sum()
    tEdgexplain = lambda3 * (T.log(T.nnet.sigmoid(c + alpha * tA * T.dot(tW, tW.transpose())))).sum()
    
    tCost = tEmbedding +  tEdgexplain + tRegularizer
    tGamma = T.scalar(name="learning_rate")
    tgrad_H, tgrad_W = T.grad(cost=tCost, wrt=[tH, tW]) 

    trainH = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tH:tH - tGamma * tgrad_H},
            name="trainH")
    trainHDirect = theano.function(
            inputs=[],
            outputs=[tCost],
            updates={tH:T.dot(T.dot( T.inv(T.dot(tW.T, tW) + lambda2 * tI ), tW.T), tX)},
            name="trainHDirect")                                   
                        
    trainW = theano.function(
            inputs=[tGamma],
            outputs=[tCost],
            updates={tW:tW - tGamma * tgrad_W},
            name="trainW")

    for i in range(0,iterations):
        if use_regularized_ridge_regression_for_h:
            tCostH = trainHDirect()
        else:
            tCostH = trainH(np.asarray(learning_rate,dtype=theano.config.floatX))
        tCostW = trainW(np.asarray(learning_rate,dtype=theano.config.floatX));
        print 'iter ' + str(i) + ':', np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        

    return tW.get_value(),tH.get_value()
if __name__=="__main__":
    print "USAGE : NMF.py <matrixDim> <latentFactors> <iter>"
    print 'input matrix X is assumed to be a square for simplicity, the algorithms work with any type of input matrix.'
    
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    it = int(sys.argv[3])
    rng = np.random
    #topic-word
    Hi = rng.random((r,n)).astype(theano.config.floatX)
    #embeddings (document-topic)
    Wi = rng.random((n,r)).astype(theano.config.floatX)
    #input matrix  (document-word)
    X = rng.random((n,n)).astype(theano.config.floatX)
    #adjacancy matrix (document-document)
    A = rng.random((n,n)).astype(theano.config.floatX)
    print " --- "
    t0 = time.time()
    W,H = NMF_edgexplain(X, r, it, A, Hi, Wi, learning_rate=0.1, alpha=10, c=0)
    t1 = time.time()
    print "Time taken by CPU : ", t1-t0
    print " --- "
    t0 = time.time()
    W,H = NMF_regularized(X, r, it, Hi, Wi, learning_rate=0.1)
    t1 = time.time()
    print "Time taken by CPU : ", t1-t0
    print " --- "
    t0 = time.time()
    W,H = NMF(X,r,it,Hi,Wi)
    t1 = time.time()
    print "Time taken by Theano : ", t1-t0
    print " --- "
    t0 = time.time()
    W,H = NMFN(X,r,it,Hi,Wi)
    t1 = time.time()
    print "Time taken by CPU : ", t1-t0