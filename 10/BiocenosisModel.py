import pygame
import random

class Plant:
    def __init__(self, surface, x, y):
        self.surface = surface
        self.color = (34, 139, 34)
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.circle(self.surface, self.color, [self.x, self.y], 10)


class Herbivore:
    def __init__(self, surface, x, y):
        self.surface = surface
        self.color = (255, 255, 0)
        self.x = x
        self.y = y
        self.speed = 3
        self.direction = random.uniform(0, 2 * math.pi)
        self.vision = 50
        self.energy = 100
        self.metabolism = 0.1

    def draw(self):
        pygame.draw.circle(self.surface, self.color, [int(self.x), int(self.y)], 10)

    def move(self, plants):
        # Реализуйте перцептрон для управления движением травоядных
        input_vector = []
        for plant in plants:
            dist = math.sqrt((self.x - plant.x)**2 + (self.y - plant.y)**2)
            if dist <= self.vision:
                input_vector.append(1)
            else:
                input_vector.append(0)

        direction = self.controller.predict(input_vector)
        self.direction += (direction - 0.5) * math.pi / 2

        dx = self.speed * math.cos(self.direction)
        dy = self.speed * math.sin(self.direction)

        new_x = self.x + dx
        new_y = self.y + dy

        # Проверка столкновений с границами окна
        if new_x < 0 or new_x > self.surface.get_width():
            self.direction = math.pi - self.direction

        if new_y < 0 or new_y > self.surface.get_height():
            self.direction = -self.direction

        # Проверка столкновения с растениями
        for plant in plants:
            dist = math.sqrt((new_x - plant.x)**2 + (new_y - plant.y)**2)
            if dist <= 20:
                plants.remove(plant)
                self.energy += 20

        self.x += dx
        self.y += dy
        self.energy -= self.metabolism

    def is_dead(self):
        return self.energy <= 0


class Predator:
    def __init__(self, surface, x, y):
        self.surface = surface
        self.color = (255, 0, 0)
        self.x = x
        self.y = y
        self.speed = 5
        self.direction = random.uniform(0, 2 * math.pi)
        self.vision = 100
        self.energy = 100
        self.metabolism = 0.5

    def draw(self):
        pygame.draw.circle(self.surface, self.color, [int(self.x), int(self.y)], 10)

    def move(self, herbivores):
        # Реализуйте перцептрон для управления движением хищников
        input_vector = []
        for herbivore in herbivores:
            dist = math.sqrt((self.x - herbivore.x)**2 + (self.y - herbivore.y)**2)
            if dist <= self.vision:
                input_vector.append(1)
            else:
                input_vector.append(0)

        if any(input_vector):
            direction = self.controller.predict(input_vector)
            self.direction += (direction - 0.5) * math.pi / 2

            dx = self.speed * math.cos(self.direction)
            dy = self.speed * math.sin(self.direction)

            new_x = self.x + dx
            new_y = self.y + dy

            # Проверка столкновений с границами окна
            if new_x < 0 or new_x > self.surface.get_width():
                self.direction = math.pi - self.direction

            if new_y < 0 or new_y > self.surface.get_height():
                self.direction = -self.direction

            # Проверка столкновения с травоядными
            for herbivore in herbivores:
                dist = math.sqrt((new_x - herbivore.x)**2 + (new_y - herbivore.y)**2)
                if dist <= 20:
                    herbivores.remove(herbivore)
                    self.energy += 50

            self.x += dx
            self.y += dy
        else:
            self.energy -= self.metabolism

    def is_dead(self):
        return self.energy <= 0
        
def main():
    pygame.init()

    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))

    pygame.display.set_caption("Bioecosystem")

    # Создание объектов растений, травоядных и хищников
    plants = [Plant(screen, random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(30)]
    herbivores = [Herbivore(screen, random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(10)]
    predators = [Predator(screen, random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(3)]

    clock = pygame.time.Clock()

    # Определение перцептронов для контроля активности травоядных и хищников
    herbivore_controller = Perceptron(30)
    predator_controller = Perceptron(10)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((255, 255, 255))

        # Обновление положения и состояния растений
        for plant in plants:
            plant.draw()

        # Обновление положения и состояния травоядных
        for herbivore in herbivores:
            herbivore.controller = herbivore_controller
            herbivore.move(plants)
            herbivore.draw()

            if herbivore.is_dead():
                herbivores.remove(herbivore)

        # Обновление положения и состояния хищников
        for predator in predators:
            predator.controller = predator_controller
            predator.move(herbivores)
            predator.draw()

            if predator.is_dead():
                predators.remove(predator)

        pygame.display.flip()

        clock.tick(30)


if __name__ == '__main__':
    main()
    
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.threshold = 0.5

    def predict(self, inputs):
        summation = np.dot(self.weights, inputs)
        if summation >= self.threshold:
            return 1
        else:
            return 0