'''
Created on 25 Jan 2016

@author: https://downhill.readthedocs.org/en/stable/
'''
import climate
import downhill
import numpy as np
import theano
import theano.tensor as TT

climate.enable_default_logging()

def rand(a, b): return np.random.randn(a, b).astype('f')

A, B, K = 20, 5, 3

# Set up a matrix factorization problem to optimize.
u = theano.shared(rand(A, K), name='u')
v = theano.shared(rand(K, B), name='v')
e = TT.sqr(TT.matrix() - TT.dot(u, v))

# Minimize the regularized loss with respect to a data matrix.
y = np.dot(rand(A, K), rand(K, B)) + rand(A, B)

downhill.minimize(
    loss=e.mean() + abs(u).mean() + (v * v).mean(),
    train=[y],
    patience=0,
    batch_size=A,                 # Process y as a single batch.
    max_gradient_norm=1,          # Prevent gradient explosion!
    learning_rate=0.1,
    monitors=(('err', e.mean()),  # Monitor during optimization.
              ('|u|<0.1', (abs(u) < 0.1).mean()),
              ('|v|<0.1', (abs(v) < 0.1).mean())),
    monitor_gradients=True)

# Print out the optimized coefficients u and basis v.
print('u =', u.get_value())
print('v =', v.get_value())