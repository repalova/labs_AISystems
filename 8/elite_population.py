import random 
from deap import algorithms, base, creator, tools

def evaluate(individual):
    x = individual[0]
    y = individual[1]
    result = 1 / (1 + x ** 2 + y ** 2)
    return result,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
elite_population = toolbox.population(n=10)
CXPB = 0.5
MUTPB = 0.2
NGEN = 1000

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

    fits = toolbox.map(toolbox.evaluate, population + elite_population)
    for fit, ind in zip(fits, population + elite_population):
        ind.fitness.values = fit

    elite_population = tools.selBest(population + elite_population, k=10)

    if gen % 100 == 0:
        print("Generation:", gen, "Best fitness:", elite_population[0].fitness.values)
        
# Метод элит
toolbox.register("select", tools.selBest, k=len(population + elite_population))
# Метод рулетки
toolbox.register("select", tools.selRoulette, k=len(population + elite_population))