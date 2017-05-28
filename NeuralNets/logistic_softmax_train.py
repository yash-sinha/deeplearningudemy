import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

def y2indicator(y,K):
	N = len(y)
	ind = np.zeros((N,K))

	for i in range(N):
		ind[i,y[i]] = 1

	return ind

X, Y = get_data()
X,Y = shuffle(X,Y)

Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest= Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W = np.random.randn(D,K)
b = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis =1, keepdims=True)

def forward(X,W,b):
	return softmax(X.dot(W) +b)

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis =1)

def classification_rate(Y,P):
	return np.mean(Y==P)

def cross_entropy(T, pY):
	return - np.mean(T*np.log(pY))

train_cost =[]
test_cost =[]

learning_rate = 0.001
for i in range(10000):
	pYtrain = forward(Xtrain, W,b)
	pyTest = forward(Xtest, W,b)

	cTrain = cross_entropy(Ytrain_ind, pYtrain)
	cTest = cross_entropy(Ytest_ind, pyTest)

	train_cost.append(cTrain)
	test_cost.append(cTest)

	W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain_ind)
	b -= learning_rate*(pYtrain- Ytrain_ind).sum(axis=0)

	if i%1000 ==0:
		print(i, cTrain, cTest)

print("Final train classification rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification rate:", classification_rate(Ytest, predict(pyTest)))

legend1, = plt.plot(train_cost, label = "train_cost")
legend2, = plt.plot(test_cost, label = "test_cost")
plt.legend([legend1, legend2])
plt.show()
