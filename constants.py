# CONFIG file

EPOCHS_PER_GENERATION = 100  # How much time until we kill them all
MAXIMUM_GENERATIONS = 200

SEXUAL_REPRODUCTION = True  # Snusnu or not snusnu

POPULATION_SENSOR_RADIUS = 1
PHEROMONE_SENSOR_RADIUS = 1

LENGTH_EXP = 6  # Maximum value of neuron constants length pow(2, LENGTH_EXP)

BLOCK_VALUE = 2  # Value of a blockage in the world grid

ACTION_THRESHOLD = 0.1  # THRESHOLD for the action to be executed, should be random but would do
MOVE_THRESHOLD = (1 - ACTION_THRESHOLD) / 2  # THRESHOLD when there is to actions to choose

MUTATION_CHANCE = 1e-5
PERMANENCE_CHANCE = 1 - MUTATION_CHANCE

GENOME_LENGTH = 8
GENOME_STRUCTURE_DICT = {
    'source_type': {
        'start': 0,
        'steps': 1
    },
    'source_id': {
        'start': 1,
        'steps': 7
    },
    'dest_type': {
        'start': 8,
        'steps': 1
    },
    'dest_id': {
        'start': 9,
        'steps': 7
    },
    'weight': {
        'start': 16,
        'steps': 16
    }
}
