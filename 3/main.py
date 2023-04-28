import tensorflow as tf

# Определение входных данных и выходов
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Создание модели многослойного перцептрона
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=1000)

# Оценка качества модели на обучающей выборке
train_loss, train_acc = model.evaluate(x_train, y_train)
print('Train accuracy:', train_acc)

# Предсказание на новых данных
x_new = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_new = model.predict(x_new)
print('Predictions:', y_new)