import numpy as np
N= 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D)) #center at -2,-2
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) #center at 2,2
ones = np.array([[1]*N]).T
T = np.array([0]*50 + [1]*50)
Xb = np.concatenate((ones,X), axis=1)

w = np.random.randn(D+1)

z = Xb.dot(w)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

Y =sigmoid(z)

def cross_entropy(T,Y):
	E=0
	for i in range(N):
		if T[i] ==1:
			E -=np.log(Y[i])
		else:
			E -= np.log(1-Y[i])

	return E

learning_rate = 0.1

for i in range(100):
	if i%10==0:
		print(cross_entropy(T,Y))

	w+= learning_rate * np.dot((T-Y).T, Xb)
	Y = sigmoid(Xb.dot(w))


print("Final w:",w)
