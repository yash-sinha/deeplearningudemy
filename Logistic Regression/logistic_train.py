import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X,Y = get_binary_data()
X,Y = shuffle(X,Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b=0

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def forward(X, W, b):
	return sigmoid(X.dot(W) +b)

def classification_rate(Y,P):
	return np.mean(Y==P)

def cross_entropy(T,pY):
	return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

train_costs =[]
test_costs =[]
learning_rate = 0.001

for i in range(10000):
	pyTrain = forward(Xtrain,W,b)
	pyTest = forward(Xtest,W,b)

	cTrain = cross_entropy(Ytrain,pyTrain)
	cTest = cross_entropy(Ytest,pyTest)

	train_costs.append(cTrain)
	test_costs.append(cTest)

	W -= learning_rate*Xtrain.T.dot(pyTrain-Ytrain)
	b -= learning_rate*(pyTrain-Ytrain).sum()

	if i%1000==0:
		print (i, cTrain, cTest)

print("Final train classification", classification_rate(Ytrain, np.round(pyTrain)))
print("Final test classification", classification_rate(Ytest, np.round(pyTest)))

legend1, = plt.plot(train_costs, label = 'train cost')
legend2, = plt.plot(test_costs, label = 'test cost')

plt.legend([legend1,legend2])
plt.show()


