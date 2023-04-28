import numpy as np
import tf as tf
import plt as plt

# Загрузка данных для обучения и тестирования
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train, x_test = x_train / 255.0, x_test / 255.0

# Создание матрицы весов для сети Хопфилда
W = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if i != j:
            W[i][j] = np.sum(x_train[:, i] * x_train[:, j])


# Функция активации для нейронов
def activation(x):
    if x >= 0:
        return 1
    else:
        return -1


# Функция восстановления поврежденных цифр
def restore_digit(digit, W):
    # Создание копии поврежденной цифры
    restored_digit = np.copy(digit)

    # Итерационное обновление значений нейронов
    for i in range(10):
        for j in range(100):
            h = np.dot(W[j], restored_digit)
            restored_digit[j] = activation(h)

    return restored_digit


# Выбор случайной поврежденной цифры для восстановления
digit_idx = np.random.randint(len(x_test))
noisy_digit = x_test[digit_idx].flatten()

# Вывод поврежденной и восстановленной цифр
print("Noisy digit:")
plt.imshow(noisy_digit.reshape(28, 28), cmap='gray')
plt.show()

restored_digit = restore_digit(noisy_digit, W)
print("Restored digit:")
plt.imshow(restored_digit.reshape(28, 28), cmap='gray')
plt.show()