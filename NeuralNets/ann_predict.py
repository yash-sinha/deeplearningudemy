import numpy as np
from process import get_data

X, Y = get_data()

X = np.array([[1,2]])
W1 = np.array([[1,1], [1,0]]) 
W2 = np.array([[0,1], [1,1]]) 


M = 2 #hidden units
D = X.shape[1]
K = 2#len(set(Y)) #0 to k-1

# W1 = np.random.randn(D,M)
b1 = np.zeros(M)
# W2 = np.random.randn(M,K)
b2 = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y,P):
	return np.mean(Y==P)

print(P_Y_given_X)
print(predictions)
#print(classification_rate(Y,predictions))