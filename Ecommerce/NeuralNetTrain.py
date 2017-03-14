import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from Ecommerce.ProccessData import get_data
from Util import y2indicator, init_weight_and_biases, softmax, classification_rate, predict

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

M = 5
D = X.shape[1]
K = len(set(Y))

# create train and test sets
Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain)
Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest)

W1, b1 = init_weight_and_biases(D, M)
W2, b2 = init_weight_and_biases(M, K)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2), Z

def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY))

train_costs = []
test_costs = []

learning_rate = 0.001
for i in range(10000):
	pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
	pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

	ctrain = cross_entropy(Ytrain_ind, pYtrain)
	ctest = cross_entropy(Ytest_ind, pYtest)
	train_costs.append(ctrain)
	test_costs.append(ctest)

	#perform gradient descent
	W2 -= learning_rate * Ztrain.T.dot(pYtrain - Ytrain_ind)
	b2 -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)
	dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain)
	W1 -= learning_rate * Xtrain.T.dot(dZ)
	b1 -= learning_rate * dZ.sum(axis=0)


	# log out costs every 1000 steps
	if i % 1000 == 0:
		print(i, ctrain, ctest)

print("Final train classification rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification rate:", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()