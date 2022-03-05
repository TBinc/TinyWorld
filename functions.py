import numpy as np
import random

from constants import GENOME_STRUCTURE_DICT, GENOME_LENGTH, MUTATION_CHANCE, PERMANENCE_CHANCE


def create_genome(genome_len: int) -> np.ndarray:
    if genome_len <= 0:
        raise ArithmeticError("genome_len has to be higher than 0")
    return np.random.randint(0, high=2 ** 32, size=genome_len, dtype=np.int64)


def create_offspring(couples: np.ndarray | list) -> np.ndarray:
    offspring = []
    for couple in couples:
        offspring.append(reproduce_genome(couple))
    return np.array(offspring)


def reproduce_genome(parents: list | np.ndarray) -> np.ndarray:
    rand_int = np.random.randint(1, GENOME_LENGTH)
    p1 = random.sample(list(parents[0].genome_int), rand_int)
    p2 = random.sample(list(parents[1].genome_int), GENOME_LENGTH - rand_int)
    return np.array(p1 + p2)


def mutate_genome(genome_list: np.ndarray) -> np.ndarray:
    t = (0, 1)
    p = (PERMANENCE_CHANCE, MUTATION_CHANCE)
    return genome_list ^ np.random.choice(t, size=(len(genome_list), GENOME_LENGTH), p=p)


def int_to_hex(number: int, padding: int = 8) -> str:
    return f"{number:0{padding}x}"


def hex_to_int(number: str) -> int:
    return int(number, 16)


def int_from_bin(bin_number: int, start: int, steps: int = 1) -> int:  # works right to left
    return bin_number >> start & (2 ** steps - 1)


def int_from_bin_reversed(bin_number: int, start: int, steps: int = 1) -> int:  # works left to right
    move = 32 - start - steps
    if steps <= 0:
        raise ArithmeticError(f'Invalid steps: {steps}')
    if move < 0:
        raise ArithmeticError(f"'start' + 'steps' has to be lower or equal to 32, it is {start + steps}")
    return bin_number >> move & (2 ** steps - 1)


def value_from_gene(gene, value_type) -> int:
    return int_from_bin_reversed(gene, **GENOME_STRUCTURE_DICT[value_type])
