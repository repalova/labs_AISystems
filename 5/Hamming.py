import numpy as np

def hopfield_recovery(input_data):
    """Восстанавливает образ с помощью сети Хопфилда"""

    # Создаем матрицу весов
    W = np.zeros((len(input_data[0]), len(input_data[0])))
    for data in input_data:
        W += np.outer(data, data)

    # Диагональные элементы матрицы весов должны быть равны 0
    np.fill_diagonal(W, 0)

    # Восстанавливаем образ
    result = input_data[0]
    new_result = result.copy()
    while not np.array_equal(result, new_result):
        result = new_result.copy()
        for i in range(len(result)):
            new_result[i] = np.sign(W[i] @ result)

    return result

input_data = np.load("python.npy")

recovered_data = hopfield_recovery(input_data)

print(recovered_data)

class HammingNetwork:
    """Сеть Хемминга"""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape[0]

        # Создаем входной и выходной слои
        self.input_layer = np.zeros(input_shape)
        self.output_layer = np.zeros(self.output_shape)

        # Задаем образ для выходного слоя
        self.output_layer[0] = recovered_data
        
    def recognize_pattern(self, input_data, threshold=5):
        """Распознает образ входных данных"""

        # Сравниваем каждый входной образ с образом выходного слоя
        for i in range(self.output_shape):
            num_mismatch = np.sum(input_data != self.output_layer[i])
            if num_mismatch <= threshold:
                return i

        # Если ни один образ не соответствует входным данным, то возвращаем -1
        return -1
        
hamming_network = HammingNetwork(input_shape=(len(input_data[0]), ))

result = hamming_network.recognize_pattern(input_data[0])

if result == -1:
    print("Образ не распознан")
else:
    print("Образ распознан как образ номер", result)