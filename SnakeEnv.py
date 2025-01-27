import pygame
import random
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

snake_speed = 15
window_x = 360
window_y = 240
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()

class CustomEnv(gym.Env):
    def __init__(self, num_fruits=1, show_gui=False):
        super().__init__()
        self.num_fruits = num_fruits
        self.show_gui = show_gui
        self.action_space = spaces.Discrete(4)
        # Specify the number of elements in the observation vector
        observation_length = 4 + 2 * num_fruits + 18  # Ensure this is correct for your case
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_length,), dtype=np.float32)
        pygame.display.set_caption('Snake AI Training')
        self.game_window = pygame.display.set_mode((window_x, window_y))
        self.fps = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None):
        self.snake_position = [180, 120]
        self.snake_body = [[180, 120], [170, 120], [160, 120]]
        self.fruit_positions = [self.random_fruit_position() for _ in range(self.num_fruits)]
        self.direction = 'RIGHT'
        self.steps_since_last_fruit = 0
        self.reward = 0
        return self.get_observation(), {}

    def step(self, action):
        self.handle_events()
        self.change_direction(action)
        self.move_snake()
        self.reward += 0.00864 * len(self.snake_body) / 4
        terminated = self.check_game_over()
        reward = self.calculate_rewards(terminated)
        self.steps_since_last_fruit += 1

        if self.steps_since_last_fruit >= 1000:
            terminated = True
            reward -= 1000 * len(self.snake_body) / 10
            self.game_over()

        self.reward += reward
        if self.show_gui:
            self.game_window.fill(black)
            self.draw_elements()
            pygame.display.update()
            self.fps.tick(snake_speed)
        return self.get_observation(), self.reward, terminated, False, {}

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def calculate_rewards(self, terminated):
        reward = 0
        current_distance_to_fruit = float('inf')
        for fruit_position in self.fruit_positions:
            distance = ((self.snake_position[0] - fruit_position[0]) ** 2 +
                        (self.snake_position[1] - fruit_position[1]) ** 2) ** 0.5
            if distance < current_distance_to_fruit:
                current_distance_to_fruit = distance

        if hasattr(self, 'previous_distance_to_fruit') and current_distance_to_fruit < self.previous_distance_to_fruit:
            reward += 0.00864 * len(self.snake_body) / 6

        self.previous_distance_to_fruit = current_distance_to_fruit

        for index, fruit_position in enumerate(self.fruit_positions):
            if self.snake_position == fruit_position:
                reward += 86.4 * len(self.snake_body) / 4
                self.fruit_positions[index] = self.random_fruit_position()
                self.previous_distance_to_fruit = float('inf')
                self.steps_since_last_fruit = 0

        if terminated:
            reward -= 86.4 * len(self.snake_body) / 8

        return reward

    def change_direction(self, action):
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        new_direction = directions[action]
        if new_direction != opposite.get(self.direction, ''):
            self.direction = new_direction

    def move_snake(self):
        move_dict = {'UP': (0, -10), 'DOWN': (0, 10), 'LEFT': (-10, 0), 'RIGHT': (10, 0)}
        new_head = [self.snake_position[0] + move_dict[self.direction][0],
                    self.snake_position[1] + move_dict[self.direction][1]]
        self.snake_body.insert(0, list(new_head))
        fruit_eaten = any(fruit_pos == new_head for fruit_pos in self.fruit_positions)
        if fruit_eaten:
            index = self.fruit_positions.index(new_head)
            self.fruit_positions[index] = self.random_fruit_position()
            self.steps_since_last_fruit = 0
        else:
            self.snake_body.pop()

        self.snake_position = new_head

    def random_fruit_position(self):
        return [random.randrange(1, ((window_x - 20) // 10)) * 10,
                random.randrange(1, ((window_y - 20) // 10)) * 10]

    def draw_elements(self):
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
        for fruit_position in self.fruit_positions:
            pygame.draw.rect(self.game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

    def check_game_over(self):
        if (self.snake_position[0] < 0 or self.snake_position[0] > window_x - 10 or
            self.snake_position[1] < 0 or self.snake_position[1] > window_y - 10 or
            self.snake_position in self.snake_body[1:]):
            return True
        return False

    def get_observation(self):
        head_x, head_y = self.snake_position

        # Distances to walls
        dist_to_walls = [
            head_x / (window_x - 10),
            ((window_x - 10) - head_x) / (window_x - 10),
            head_y / (window_y - 10),
            ((window_y - 10) - head_y) / (window_y - 10)
        ]

        # Positions of fruits relative to the snake's head
        fruit_relative_positions = [
            ((fruit_position[0] - head_x) / window_x,
             (fruit_position[1] - head_y) / window_y)
            for fruit_position in self.fruit_positions
        ]

        # Find the nearest fruit
        closest_fruit = min(self.fruit_positions,
                            key=lambda x: ((x[0] - head_x) ** 2 + (x[1] - head_y) ** 2) ** 0.5)
        fruit_direction = [
            (closest_fruit[0] - head_x) / abs(closest_fruit[0] - head_x) if closest_fruit[0] != head_x else 0,
            (closest_fruit[1] - head_y) / abs(closest_fruit[1] - head_y) if closest_fruit[1] != head_y else 0
        ]

        # Direction vector
        direction_vector = [int(self.direction == d) for d in ['UP', 'DOWN', 'LEFT', 'RIGHT']]

        # Collision sensors
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        sensor_distances = [10, 20, 30]  # Different distances for sensors
        collision_sensors = []

        for direction in directions:
            dx, dy = 0, 0
            if direction == 'UP':
                dx, dy = 0, -10
            elif direction == 'DOWN':
                dx, dy = 0, 10
            elif direction == 'LEFT':
                dx, dy = -10, 0
            elif direction == 'RIGHT':
                dx, dy = 10, 0

            for distance in sensor_distances:
                sensor_x = head_x + dx * (distance // 10)
                sensor_y = head_y + dy * (distance // 10)
                collision_sensors.append(1 if [sensor_x, sensor_y] in self.snake_body else 0)

        # Combine all observation components
        observation = dist_to_walls + \
                      [pos for sublist in fruit_relative_positions for pos in sublist] + \
                      direction_vector + \
                      fruit_direction + \
                      collision_sensors

        return np.array(observation, dtype=np.float32)

    def game_over(self):
        my_font = pygame.font.SysFont('times new roman', 15)
        game_over_surface = my_font.render('YOU ARE DEAD ' + str(self.reward), True, red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (window_x / 2, window_y / 4)
        if self.show_gui:
            self.game_window.blit(game_over_surface, game_over_rect)
            pygame.display.flip()
