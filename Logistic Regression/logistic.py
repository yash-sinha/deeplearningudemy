import numpy as np
N= 100
D = 2

x = np.random.randn(N,D)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,x), axis=1)

w = np.random.randn(D+1)

z = Xb.dot(w)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

print(sigmoid(z))