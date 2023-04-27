import random

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=1):
        """
        distances: distance matrix of shape (n_cities, n_cities)
        n_ants: number of ants
        n_iterations: number of iterations
        decay: rate of pheromone decay
        alpha: relative weight of pheromone in the ant decision rule
        beta: relative weight of distance in the ant decision rule
        """
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        n_cities = len(self.distances)
        pheromones = [[1 / (n_cities * n_cities) for j in range(n_cities)] for i in range(n_cities)] # initialize pheromones
        best_path = None
        best_path_length = float('inf')

        for it in range(self.n_iterations):
            paths = []
            path_lengths = []
            for ant in range(self.n_ants):
                path = self._create_path(pheromones, self.distances)
                path_length = self._calculate_path_length(path, self.distances)
                paths.append(path)
                path_lengths.append(path_length)
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            pheromones = self._update_pheromones(pheromones, paths, path_lengths)
            pheromones = self._decay_pheromones(pheromones, self.decay)

        return best_path, best_path_length

    def _create_path(self, pheromones, distances):
        n_cities = len(distances)
        start_city = random.randint(0, n_cities - 1)
        path = [start_city]

        while len(path) < n_cities:
            current_city = path[-1]
            next_city = self._choose_next_city(current_city, path, pheromones, distances)
            path.append(next_city)

        return path

    def _choose_next_city(self, current_city, path, pheromones, distances):
        unvisited_cities = [i for i in range(len(pheromones)) if i not in path]
        total = 0
        probabilities = []
        for city in unvisited_cities:
            pheromone = pheromones[current_city][city] ** self.alpha
            distance = distances[current_city][city] ** self.beta
            total += pheromone * distance
            probabilities.append((city, total))

        rand = random.uniform(0, total)
        for city, p in probabilities:
            if rand < p:
                return city

    def _calculate_path_length(self, path, distances):
        length = 0
        for i in range(len(path)):
            length += distances[path[i - 1]][path[i]]
        return length

    def _update_pheromones(self, pheromones, paths, path_lengths):
        for i in range(len(pheromones)):
            for j in range(len(pheromones)):
                pheromones[i][j] *= self.decay
                for path, path_length in zip(paths, path_lengths):
                    if (i, j) in zip(path, path[1:]):
                        pheromones[i][j] += 1 / path_length
        return pheromones

    def _decay_pheromones(self, pheromones, decay):
        n_cities = len(pheromones)
        for i in range(n_cities):
            for j in range(n_cities):
                pheromones[i][j] *= decay
        return pheromones
        
from itertools import permutations

# calculate distance matrix
n_cities = 10
positions = [(random.uniform(-1000, 1000), random.uniform(-1000, 1000)) for _ in range(n_cities)]
distances = [[((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5 for b in positions] for a in positions]

# brute-force solution (for testing)
best_path, best_path_length = min([(p, sum([distances[p[i - 1]][p[i]] for i in range(len(p))])) for p in permutations(range(n_cities))], key=lambda x: x[1])

# ant colony optimization
ant_colony = AntColony(distances, n_ants=50, n_iterations=100, decay=0.5)
path, length = ant_colony.run()

print(path)
print(length)
print(best_path, best_path_length)