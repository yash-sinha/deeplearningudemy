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

M =5
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest= Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D,M)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis =1, keepdims=True)

def forward(X,W1,b1, W2, b2):
	Z = np.tanh(X.dot(W1)+b1)
	return softmax(Z.dot(W2) +b2), Z

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
	pYtrain, Ztrain = forward(Xtrain, W1,b1, W2, b2)
	pyTest, Ztest = forward(Xtest, W1,b1, W2, b2)

	cTrain = cross_entropy(Ytrain_ind, pYtrain)
	cTest = cross_entropy(Ytest_ind, pyTest)

	train_cost.append(cTrain)
	test_cost.append(cTest)

	W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
	b2 -= learning_rate*(pYtrain- Ytrain_ind).sum()
	dZ = (pYtrain-Ytrain_ind).dot(W2.T) * (1-Ztrain*Ztrain)

	W1 -= learning_rate*Xtrain.T.dot(dZ)
	b1 -= learning_rate*dZ.sum(axis=0)

	if i%1000 ==0:
		print(i, cTrain, cTest)

print("Final train classification rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification rate:", classification_rate(Ytest, predict(pyTest)))

legend1, = plt.plot(train_cost, label = "train_cost")
legend2, = plt.plot(test_cost, label = "test_cost")
plt.legend([legend1, legend2])
plt.show()


