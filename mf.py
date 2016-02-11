'''
Created on 3 Feb 2016

@author: downloaded from http://un-mindlab.blogspot.com.au/2013/07/matrix-factorization-with-theano.html
To measure this difference in context, a gradient descent version of Juan C. Caicedo matrix factorization algorithm was implemented
'''
import theano
import theano.tensor as Th
import numpy as np
import time
import sys

def MFmanual(V,T,r,l,gamma,iterations,P=None,Q=None,H=None):
    """
        Paramters:
         V : As many rows as documents
         T : As many rows as documents
        """
    V = V.T
    T = T.T
    rng = np.random
    n = np.size(V,1)
    td = np.size(T,0)
    vd = np.size(V,0)
    if(P is None):
        P = rng.random((vd,r)).astype(theano.config.floatX)
    if(Q is None):
        Q = rng.random((td,r)).astype(theano.config.floatX)
    if(H is None):
        H = rng.random((r,n)).astype(theano.config.floatX)


    tV = theano.shared(V.astype(theano.config.floatX),name="V")
    tT = theano.shared(T.astype(theano.config.floatX),name="T")
    tH = theano.shared(H,name="H")
    tQ = theano.shared(Q,name="Q")
    tP = theano.shared(P,name="P")
    tLambda = Th.scalar(name="l")
    tGamma = Th.scalar(name="gamma")

    tEV = (1.0/2.0)*((tV-Th.dot(tP,tH))**2).sum() 
    tET = (1.0/2.0)*((tT-Th.dot(tQ,tH))**2).sum() 
    tReg = (1.0/2.0)*tLambda*(((tP**2).sum())+((tQ**2).sum())+((tH**2).sum()))

    tCost = tEV + tET + tReg

    gP = -1.0 *(Th.dot(tV,tH.T) - Th.dot(tP,Th.dot(tH,tH.T)) - tLambda*tP)
    gQ = -1.0 *(Th.dot(tT,tH.T) - Th.dot(tQ,Th.dot(tH,tH.T)) - tLambda*tQ)
    gH = -1.0 *(Th.dot(tP.T,tV) - Th.dot(tP.T,Th.dot(tP,tH)) + Th.dot(tQ.T,tT) - Th.dot(tQ.T,Th.dot(tQ,tH)) - tLambda*tH)


    train = theano.function(
            inputs=[tGamma,tLambda],
            outputs=[theano.Out(tCost,borrow=True)],
            updates={tP:tP - tGamma * gP, tQ : tQ - tGamma*gQ, tH : tH - tGamma*gH },
            name="train")

    for i in range(0,iterations):
        print train(np.asarray(gamma,dtype=theano.config.floatX),np.asarray(l,dtype=theano.config.floatX));

    return tP.get_value(),tQ.get_value(),tH.get_value()
def MFauto(V,T,r,l,gamma,iterations,P=None,Q=None,H=None):
    """
        Paramters:
         V : As many rows as documents
         T : As many rows as documents
        """
    V = V.T
    T = T.T
    rng = np.random
    n = np.size(V,1)
    td = np.size(T,0)
    vd = np.size(V,0)
    if(P is None):
        P = rng.random((vd,r)).astype(theano.config.floatX)
    if(Q is None):
        Q = rng.random((td,r)).astype(theano.config.floatX)
    if(H is None):
        H = rng.random((r,n)).astype(theano.config.floatX)


    tV = theano.shared(V.astype(theano.config.floatX),name="V")
    tT = theano.shared(T.astype(theano.config.floatX),name="T")
    tH = theano.shared(H,name="H")
    tQ = theano.shared(Q,name="Q")
    tP = theano.shared(P,name="P")
    tLambda = Th.scalar(name="l")
    tGamma = Th.scalar(name="gamma")

    tEV = (1.0/2.0)*((tV-Th.dot(tP,tH))**2).sum() 
    tET = (1.0/2.0)*((tT-Th.dot(tQ,tH))**2).sum() 
    tReg = (1.0/2.0)*tLambda*(((tP**2).sum())+((tQ**2).sum())+((tH**2).sum()))

    tCost = tEV + tET + tReg

    gP,gQ,gH = Th.grad(tCost, [tP, tQ, tH])  

    train = theano.function(
            inputs=[tGamma,tLambda],
            outputs=[theano.Out(tCost,borrow=True)],
            updates={tP:tP - tGamma * gP, tQ : tQ - tGamma*gQ, tH : tH - tGamma*gH },
            name="train")

    for i in range(0,iterations):
        print train(np.asarray(gamma,dtype=theano.config.floatX),np.asarray(l,dtype=theano.config.floatX));

    return tP.get_value(),tQ.get_value(),tH.get_value()
    
if __name__=="__main__":
    print "USAGE : MF.py   "
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    it = int(sys.argv[3])
    rng = np.random
    V = rng.random((n,n)).astype(theano.config.floatX)
    T = rng.random((n,n)).astype(theano.config.floatX)
    P = rng.random((n,r)).astype(theano.config.floatX)
    Q = rng.random((n,r)).astype(theano.config.floatX)
    H = rng.random((r,n)).astype(theano.config.floatX)
    t0 = time.time()
    MFauto(V,T,r,0.1,1e-5,it,P,Q,H)
    t1 = time.time()
    print "Time taken by Theano's automatic gradient : ", t1-t0
    t0 = time.time()
    MFmanual(V,T,r,0.1,1e-5,it,P,Q,H)
    t1 = time.time()
    print "Time taken by Theano's manual gradient : ", t1-t0