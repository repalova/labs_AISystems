import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs) * 2 - 1

    def predict(self, inputs):
        return np.dot(inputs, self.weights)

    def get_weights(self):
        return self.weights
        
class MLP:
    def __init__(self, input_shape, hidden_shape, output_shape):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.weights = np.random.rand(input_shape * hidden_shape +
                                      hidden_shape * output_shape) * 2 - 1

    def forward(self, inputs):
        input_weights = self.weights[:self.input_shape * self.hidden_shape].reshape(self.input_shape, self.hidden_shape)
        hidden_weights = self.weights[self.input_shape * self.hidden_shape:].reshape(self.hidden_shape, self.output_shape)

        hidden = np.dot(inputs, input_weights)
        hidden = self.activation_func(hidden)

        output = np.dot(hidden, hidden_weights)
        output = self.activation_func(output)

        return output

    def activation_func(self, x):
        return 1 / (1 + np.exp(-x))

    def get_weights(self):
        return self.weights
        
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_prob):
        self.population_size = population_size
        self.mutation_prob = mutation_prob

    def init_population(self, pop_shape):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.rand(*pop_shape) * 2 - 1
            population.append(chromosome)

        return population

    def fitness(self, X, y, chromosome):
        accuracy = 0
        for i, x in enumerate(X):
            perceptron = Perceptron(len(x))
            perceptron.weights = chromosome

            prediction = perceptron.predict(x)
            if (prediction >= 0 and y[i] == 1) or (prediction < 0 and y[i] == 0):
                accuracy += 1

        accuracy /= len(X)
        return accuracy

    def inverse_loss(self, X, y, chromosome, mlp):
        loss = 0
        for i, x in enumerate(X):
            output = mlp.forward(x)
            error = output - y[i]

            loss += error ** 2

        return 1 / loss

    def crossover(self, chromosome1, chromosome2):
        split = np.random.randint(0, len(chromosome1) + 1)

        new_chromosome = np.concatenate([chromosome1[:split], chromosome2[split:]])

        return new_chromosome

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_prob:
                chromosome[i] = np.random.rand() * 2 - 1

        return chromosome

    def train_perceptron(self, X_train, y_train, X_test, y_test):
        population = self.init_population(pop_shape=(len(X_train[0]), ))

        for _ in range(50):
            population_fitness = [(chromosome, self.fitness(X_train, y_train, chromosome)) for chromosome in population]
            population_fitness = sorted(population_fitness, key=lambda x: x[1])[:int(self.population_size * 0.2)]

            new_population = []
            for i in range(int(self.population_size * 0.8)):
                parent1 = population_fitness[np.random.randint(len(population_fitness))][0]
                parent2 = population_fitness[np.random.randint(len(population_fitness))][0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = population_fitness + [(chromosome, self.fitness(X_train, y_train, chromosome)) for chromosome in
                                               new_population]
            population = sorted(population, key=lambda x: x[1], reverse=True)[:self.population_size]

        best_chromosome = population[0][0]

        perceptron = Perceptron(len(X_train[0]))
        perceptron.weights = best_chromosome

        y_pred = [1 if perceptron.predict(x) >= 0 else 0 for x in X_test]

        return accuracy_score(y_test, y_pred)

    def train_mlp(self, X_train, y_train, X_test, y_test):
        mlp = MLP(input_shape=len(X_train[0]), hidden_shape=5, output_shape=1)

        for _ in range(50):
            population = self.init_population(pop_shape=(mlp.input_shape * mlp.hidden_shape + mlp.hidden_shape * mlp.output_shape, ))
            population_fitness = [(chromosome, self.inverse_loss(X_train, y_train, chromosome, mlp)) for chromosome in population]
            population_fitness = sorted(population_fitness, key=lambda x: x[1], reverse=True)[:int(self.population_size * 0.2)]

            new_population = []
            for i in range(int(self.population_size * 0.8)):
                parent1 = population_fitness[np.random.randint(len(population_fitness))][0]
                parent2 = population_fitness[np.random.randint(len(population_fitness))][0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = population_fitness + [(chromosome, self.inverse_loss(X_train, y_train, chromosome, mlp)) for chromosome in
                                               new_population]
            population = sorted(population, key=lambda x: x[1], reverse=True)[:self.population_size]

            best_chromosome = population[0][0]
            mlp.weights = best_chromosome

        return mlp
        
ga = GeneticAlgorithm(population_size=20, mutation_prob=0.1)
accuracy = ga.train_perceptron(X_train, y_train, X_test, y_test)
print(f"Accuracy of Perceptron: {accuracy:.2f}")

ga = GeneticAlgorithm(population_size=20, mutation_prob=0.1)
mlp = ga.train_mlp(X_train, y_train, X_test, y_test)

y_pred = [1 if mlp.forward(x) >= 0.5 else 0 for x in X_test]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of MLP: {accuracy:.2f}")
