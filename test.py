import cPickle as pickle
import numpy as np

a = np.zeros(100)
A = np.ones(10)
pickle.dump([a, A], open('test.dat', 'wb'), True)
[b, B] = pickle.load(open('test.dat', 'r'))
print b[99], B[9]