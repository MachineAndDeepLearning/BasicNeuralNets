import numpy as np

from Ecommerce.ProccessData import get_data
from Util import softmax, init_weight_and_biases, classification_rate

X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))

W1, b1 = init_weight_and_biases(D, M)
W2, b2 = init_weight_and_biases(M, K)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis=1)

print("Classification rate: ", classification_rate(Y, predictions))


