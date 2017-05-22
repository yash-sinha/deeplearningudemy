import numpy as np
import matplotlib.pyplot as plt
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

w = np.array([0,4,4]) #variance =1 for numpy randn 0 = bias 
#y =-x

plt.scatter(X[:,0], X[:,1], c = T, s=100, alpha = 0.5) #c = color

x_axis = np.linspace(-6,-6,100) #-6,-6 range, 100 points
y_axis = -x_axis
plt.plot(x_axis,y_axis)
plt.show()