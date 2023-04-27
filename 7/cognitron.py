from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

X_train = X[:100] / 255.0
y_train = y[:100]

import numpy as np

class Cognitron:
    """Когнитрон для распознавания произвольных образов"""

    def __init__(self, input_shape, receptive_field_size=5, num_receptive_fields=4, num_iterations=50):
        self.input_shape = input_shape
        self.receptive_field_size = receptive_field_size
        self.num_receptive_fields = num_receptive_fields
        self.num_iterations = num_iterations

        # Создаем случайные веса для сверточных ядер
        self.filters = np.random.randn(num_receptive_fields, receptive_field_size, receptive_field_size)

    def process_input(self, input_image):
        # Преобразуем двумерное входное изображение в трехмерное
        input_image = input_image.reshape(self.input_shape)

        # Создаем массив размером с основное изображение, заполненный -1
        result = np.zeros_like(input_image) - 1

        # Проходим сверткой по каждому ядру
        for r in range(self.num_receptive_fields):
            filter_response = np.zeros_like(input_image)
            for x in range(self.receptive_field_size // 2, input_image.shape[1] - self.receptive_field_size // 2):
                for y in range(self.receptive_field_size // 2, input_image.shape[0] - self.receptive_field_size // 2):
                    # Проход сверткой
                    convolution = (input_image[y - self.receptive_field_size // 2 : y + self.receptive_field_size // 2 + 1,
                                               x - self.receptive_field_size // 2 : x + self.receptive_field_size // 2 + 1] * self.filters[r]).sum()
                    filter_response[y, x] = convolution

            # Обновляем веса сверточного ядра
            self.filters[r] += 0.1 * (input_image - np.sign(filter_response))[:, :, np.newaxis] * self.filters[r]

            # Получаем результат
            result[filter_response >= 0] = 1

        return result.flatten()

    def fit(self, X):
        # Обучаем когнитрон на входных данных
        for i in range(self.num_iterations):
            for j in range(X.shape[0]):
                self.process_input(X[j])
                
    cognitron = Cognitron(input_shape=(28, 28), receptive_field_size=5, num_receptive_fields=4, num_iterations=50)
cognitron.fit(X_train)

plt.imshow(X_train[4].reshape(28, 28), cmap="binary")
plt.show()

result = cognitron.process_input(X_train[4])
print(result.reshape(28, 28))