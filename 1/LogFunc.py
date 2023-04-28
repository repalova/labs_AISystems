import numpy as np

# Однонейронный перцептрон
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    # Функция активации (ступенька)
    def activation(self, x):
        return 1 if x > 0 else 0

    # Предсказание выходных данных
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        a = self.activation(z)
        return a

    # Обучение перцептрона
    def train(self, x, y, epochs=10, lr=1):
        for epoch in range(epochs):
            for i in range(len(x)):
                z = np.dot(x[i], self.weights) + self.bias
                a = self.activation(z)
                error = y[i] - a
                self.weights += lr * error * x[i]
                self.bias += lr * error

# Обучение перцептрона на логической функции "и"
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X, y)

# Проверка работы перцептрона на логической функции "и"
print(perceptron_and.predict(np.array([0, 0]))) # 0
print(perceptron_and.predict(np.array([0, 1]))) # 0
print(perceptron_and.predict(np.array([1, 0]))) # 0
print(perceptron_and.predict(np.array([1, 1]))) # 1

# Обучение перцептрона на логической функции "или"
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X, y)

# Проверка работы перцептрона на логической функции "или"
print(perceptron_or.predict(np.array([0, 0]))) # 0
print(perceptron_or.predict(np.array([0, 1]))) # 1
print(perceptron_or.predict(np.array([1, 0]))) # 1
print(perceptron_or.predict(np.array([1, 1]))) # 1

# Обучение перцептрона на логической функции "не"
X = np.array([[0], [1]])
y = np.array([1, 0])
perceptron_not = Perceptron(input_size=1)
perceptron_not.train(X, y)

# Проверка работы перцептрона на логической функции "не"
print(perceptron_not.predict(np.array([0]))) # 1
print(perceptron_not.predict(np.array([1]))) # 0