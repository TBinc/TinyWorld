# A grid that manages the positions and interactions of the creatures
import numpy as np

from constants import GENOME_LENGTH, POPULATION_SENSOR_RADIUS, PHEROMONE_SENSOR_RADIUS, BLOCK_VALUE, \
    EPOCHS_PER_GENERATION, SEXUAL_REPRODUCTION, MAXIMUM_GENERATIONS
from creature import Creature
from functions import create_offspring, mutate_genome
from structs import Coordinate, Direction
import random


# TODO: MAKE ALL THE CREATURES CALCULATION AT THE SAME TIME IN AN ARRAY

class World:
    def __init__(self, size: tuple = (100, 100), population: int = 1000):
        self.size: Coordinate = Coordinate._make(size)
        self.generation = 0
        self.grid: np.ndarray = np.full(size, 0, dtype=np.dtype('O'))
        self.pheromone_grid: np.ndarray = np.full(size, 0, dtype=np.int8)
        self.population: int = population
        self.epoch: int = 0
        self.creatures: np.ndarray = self._create_creatures()
        self._max_possible_distance: int = max(self.size)
        self._block_count: int = 1
        self._kill_mask: np.ndarray = self._create_circular_mask(radius=min(size) // 4)
        #  self._kill_mask: np.ndarray = self._create_circular_mask(center=Coordinate(x=2, y=2), radius=min(size) // 3)
        self._total_grid: list = []

    def __str__(self):
        return np.array2string(self.grid, separator=' ')  # This will show all the creatures

    def _create_creatures(self, first_generation: bool = True, genome_int: np.ndarray = None) -> np.ndarray:
        creature_array = []
        for pop in range(self.population):
            while self.grid[new_pos := (Coordinate(
                    x=np.random.randint(0, self.size.x),
                    y=np.random.randint(0, self.size.y)))] != 0:
                pass

            if not first_generation:
                gi = genome_int[pop]
            else:
                gi = None

            c = Creature(
                position=new_pos,
                genome_length=GENOME_LENGTH,
                genome_int=gi,
                world=self,
                first_generation=first_generation)

            self.grid[new_pos] = c

            creature_array.append(c)

        return np.array(creature_array)

    def _calculate_step(self):
        """
        Updates every creature and then moves them in the grid
        :return:
        """
        for creature in self.creatures:
            creature.calculate_step()

        for creature in self.creatures:
            self.grid[creature.position] = 0
            creature.move()
            self.grid[creature.next_position] = creature

    def _isolation_mask(self):  # If there is more than 4 creatures in a 3*3 square, kill them
        MAX_CREATURES = 3


    def _kill_them(self, kill_mask: np.ndarray):
        creatures_to_kill = [c for c in self.grid[kill_mask] if isinstance(c, Creature)]
        for creature in creatures_to_kill:
            creature.is_alive = False  # Muahaha

    def _reproduce_alive(self):
        survivors = np.array([creature for creature in self.creatures if creature.is_alive])

        if SEXUAL_REPRODUCTION:
            couples = [random.sample(list(survivors), 2) for _ in range(self.population)]
            offspring_genome = create_offspring(couples)
        else:
            offspring_genome = survivors

        return mutate_genome(offspring_genome)

    def run(self):
        while self.generation < MAXIMUM_GENERATIONS:
            self._total_grid.append([])
            self._total_grid[-1].append(np.where(self.grid == 0, self.grid, 3).astype(int))
            while self.epoch < EPOCHS_PER_GENERATION:
                self._calculate_step()
                self._total_grid[-1].append(np.where(self.grid == 0, self.grid, 3).astype(int))  #  Save the grid
                if self.epoch + 1 == EPOCHS_PER_GENERATION:
                    print(f"""Generation: {self.generation}\nEpoch: {self.epoch}\nPopulation: {np.count_nonzero(self.grid)}\n{self}\n{'-' * 40}""")
                self.epoch += 1
            self._kill_them(self._kill_mask)
            genome_sons = self._reproduce_alive()

            self.grid = np.full(self.size, 0, dtype=np.dtype('O'))
            self.creatures = self._create_creatures(first_generation=False, genome_int=genome_sons)

            self.generation += 1
            self.epoch = 0

        np.save('grid', np.array(self._total_grid))

    def coordinate_is_in_bounds(self, coordinate: Coordinate):
        return not (
                coordinate.x < 0 or
                coordinate.x >= self.size[1] or
                coordinate.y < 0 or
                coordinate.y >= self.size[0]
        )

    def _create_circular_mask(self, center: Coordinate = None, radius: int = None):
        h = self.size[0]
        w = self.size[1]

        # if h % 2 == 0: h = h + 1
        # if w % 2 == 0: w = w + 1

        if center is None:  # use the middle of the array
            center = Coordinate._make((int(w / 2), int(h / 2)))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center.x, center.y, w - center.y, h - center.x)

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center.x) ** 2 + (y - center.y) ** 2)

        mask = dist_from_center <= radius
        return mask

    def place_pheromone(self, pheromone_level: int, position: Coordinate):  # Place qty in position and half around it
        self.pheromone_grid[position] += pheromone_level // 2
        for direction in Direction:
            next_pos = position + direction.get_coordinate()
            if self.coordinate_is_in_bounds(next_pos):
                self.pheromone_grid[next_pos] += pheromone_level // 2

    def pheromone_on_direction(self, position: Coordinate, direction: Direction):
        if direction == Direction.C:
            direction = Direction.N

        original_pos = position
        pheromone_val = 0
        while self.coordinate_is_in_bounds(position := position + direction.get_coordinate()):
            pheromone_val += self.pheromone_grid[position]

        position = original_pos
        while self.coordinate_is_in_bounds(position := position - direction.get_coordinate()):
            pheromone_val -= self.pheromone_grid[position]

        return 2 * pheromone_val / self._max_possible_distance

    def pheromone_density(self, position: Coordinate):
        mask = self._create_circular_mask(position, PHEROMONE_SENSOR_RADIUS)
        return np.sum(self.pheromone_grid * mask) / np.sum(mask)

    def blocks_on_direction(self, position: Coordinate, direction: Direction):
        if direction == Direction.C:
            direction = Direction.N  # TODO: There should be a better way to address Direction.C

        original_pos = position
        block_val = 0
        while self.coordinate_is_in_bounds(position := position + direction.get_coordinate()):
            if self.grid[position] == BLOCK_VALUE:
                block_val += 1

        position = original_pos
        while self.coordinate_is_in_bounds(position := position - direction.get_coordinate()):
            if self.grid[position] == BLOCK_VALUE:
                block_val -= 1

        return 2 * block_val / self._max_possible_distance

    def population_on_direction(self, position: Coordinate, direction: Direction):
        if direction == Direction.C:
            direction = Direction.N

        original_pos = position
        population_val = 0
        while self.coordinate_is_in_bounds(position := position + direction.get_coordinate()):
            if isinstance(self.grid[position], Creature):
                population_val += 1

        position = original_pos
        while self.coordinate_is_in_bounds(position := position - direction.get_coordinate()):
            if isinstance(self.grid[position], Creature):
                population_val -= 1

        return 2 * population_val / self._max_possible_distance

    def block_next_on_direction(self, position: Coordinate, direction: Direction, max_distance: int):
        steps = 1
        val = 0
        while self.coordinate_is_in_bounds(
                position := position + direction.get_coordinate()
        ) and (steps := steps + 1) <= max_distance:
            if self.grid[position] == BLOCK_VALUE:
                val += 1

        return val / steps

    def population_density(self, position: Coordinate):
        mask = self._create_circular_mask(position, radius=POPULATION_SENSOR_RADIUS)
        m = np.ma.array(self.grid, mask=1 - mask)
        return len(np.ma.MaskedArray.nonzero(m)) / np.sum(mask)

    def population_next_creature_on_direction(self, position: Coordinate, direction: Direction, max_distance: int):
        steps = 1
        val = 0
        while (steps := steps + 1) <= max_distance \
                and self.coordinate_is_in_bounds(position := position + direction.get_coordinate()):
            if isinstance(self.grid[position], Creature):
                val += 1

        return val / steps

    def distance_to_border(self, position: Coordinate, direction: Direction):
        if direction == Direction.C:
            direction = Direction.N

        coord = direction.get_coordinate()
        d = 1

        for axis in ('x', 'y'):
            actual_dir = getattr(coord, axis)
            if actual_dir > 0:
                d *= (getattr(self.size, axis) - getattr(position, axis))
            elif actual_dir < 0:
                d *= getattr(position, axis)

        return d / (self.size.x * self.size.y)

    def nearest_border_distance(self, position: Coordinate):
        return min(
            position.x / self.size.x,
            position.y / self.size.y,
            (self.size.x - position.x) / self.size.x,
            (self.size.y - position.y) / self.size.y)
