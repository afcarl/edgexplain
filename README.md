Comparing several algorithms for non-negative matrix factorisation:

1- NMFN: numpy implementation with multiplicative update rules (copied from http://un-mindlab.blogspot.com/)

2- NMF: Theano implementation with multiplicative update rules (copied from http://un-mindlab.blogspot.com/)

3- NMF_regularized: Theano implementation with coordinate descent (regularised)

4- NMF_edgexplain: Theano implementation with coordinate descent (regularised with edgexplain term)



USAGE : Python nmf.py <matrixDim> <latentFactors> <iter>

The input matrix X is assumed to be a square matrix for simplicity, the algorithms work with any type of input matrix regardless.


Example: python nmf.py 1000 100 50



