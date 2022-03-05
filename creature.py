from dataclasses import dataclass
from typing import Any

import numpy as np
from functions import create_genome, int_to_hex
from structs import Coordinate, Direction
from brain import Brain


@dataclass
class Creature:
    world: Any
    genome_length: int = 8
    position: Coordinate = Coordinate(0, 0)
    is_alive: bool = True
    max_distance_per_step: int = 1
    pheromone: int = 1
    age: int = 0
    oscillator_period: int = 12
    direction: Direction = Direction(np.random.randint(1, 10))
    long_probe_distance: int = 32
    responsiveness: int = 16  # makes it all shaky, useless for now
    first_generation: bool = True
    genome_int: np.ndarray = None

    def __post_init__(self):
        if self.first_generation:
            self.genome_int = create_genome(self.genome_length)

        self.genome_hex: np.ndarray = np.array([int_to_hex(x) for x in self.genome_int])

        direction = Direction.get_random_direction()

        self.direction = direction

        self.brain = Brain(self.genome_int, self)

        self.previous_position: Coordinate = self.position
        self.next_position: Coordinate = self.position

    def __str__(self):
        return f"Creature\nAlive: {self.is_alive}\nPosition: {self.position}\nGenome: {self.genome_hex}"

    def __repr__(self):  # This will show the creature on the world
        return "1"

    def __or__(self, other: int):
        return False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def emit_pheromone(self, pheromone_level):
        self.world.place_pheromone(pheromone_level, self.position)

    def genetic_similarity_with_neighbor(self):
        next_position = self.position + self.direction.get_coordinate()
        if not self.world.coordinate_is_in_bounds(next_position):
            return 0.5

        if not isinstance(self.world.grid[next_position], Creature):
            return 0.5

        genome_diff_array = [bin(x).count('1') / 32 for x
                             in (self.genome_int ^ self.world.grid[next_position].genome_int)]

        return np.mean(genome_diff_array)

    def calculate_step(self):
        next_move = self.position + self.brain.calculate_step()
        if self.world.coordinate_is_in_bounds(next_move):
            self.next_position = self.position if self.world.grid[next_move] != 0 else next_move

    def move(self):
        general_direction = self.next_position - self.position
        self.direction = Direction.from_coordinate(general_direction.get_ones())

        while self.world.grid[self.next_position] != 0:
            self.next_position += self.direction.rotate(4).get_coordinate()
            if not self.world.coordinate_is_in_bounds(self.next_position):
                self.next_position = self.position
                break

        self.previous_position = self.position
        self.position = self.next_position
        self.age += 1




