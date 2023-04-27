import random
import math

def simulated_annealing(function, initial_state, initial_temperature, cooling_rate, stopping_temperature):
    current_state = initial_state
    current_cost = function(*current_state)

    temperature = initial_temperature

    while temperature > stopping_temperature:
        new_state = get_neighbour(current_state)
        new_cost = function(*new_state)

        delta_cost = new_cost - current_cost

        if delta_cost > 0 or math.exp(delta_cost / temperature) > random.random():
            current_state = new_state
            current_cost = new_cost

        temperature *= cooling_rate

    return current_state, current_cost

def get_neighbour(point, scale=2):
    return [x + random.uniform(-scale, scale) for x in point]


def cost_function(x, y):
    return 1 / (1 + x**2 + y**2)

# определяем начальную точку и другие параметры алгоритма
initial_state = [0, 0]
initial_temperature = 100
cooling_rate = 0.99
stopping_temperature = 1e-8

# запускаем алгоритм и выводим результаты
result = simulated_annealing(cost_function, initial_state, initial_temperature, cooling_rate, stopping_temperature)
print(f"Максимум функции {cost_function(*result[0])} достигается в точке x={result[0][0]}, y={result[0][1]}")