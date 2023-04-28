import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append(2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1)
            self.weights.append(2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                activations = [X[i]]
                for j in range(len(self.weights)):
                    activation = sigmoid(np.dot(activations[-1], self.weights[j]))
                    activations.append(activation)
                error = y[i] - activations[-1]
                deltas = [error * sigmoid_derivative(activations[-1])]
                for k in range(len(activations) - 2, 0, -1):
                    delta = np.dot(deltas[-1], self.weights[k].T) * sigmoid_derivative(activations[k])
                    deltas.append(delta)
                deltas.reverse()
                for j in range(len(self.weights)):
                    layer = np.atleast_2d(activations[j])
                    delta = np.atleast_2d(deltas[j])
                    self.weights[j] += learning_rate * np.dot(layer.T, delta)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        activations = [X]
        for j in range(len(self.weights)):
            activation = sigmoid(np.dot(activations[-1], self.weights[j]))
            activations.append(activation)
        return activations[-1]

# загружаем данные
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

# preprocess data
X = X / 16.0
y = np.eye(10)[y]

# обучаем сеть
nn = NeuralNetwork([64, 16, 10])
nn.fit(X, y)

# оценим производительность
from sklearn.metrics import accuracy_score
y_pred = np.argmax(nn.predict(X), axis=1)
y_true = np.argmax(y, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))


from sklearn.linear_model import Perceptron

# обучим перцетрон
clf = Perceptron()
clf.fit(X, y_true)

# оценим производительность
y_pred = clf.predict(X)
print("Perceptron accuracy:", accuracy_score(y_true, y_pred))