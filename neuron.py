from enum import Enum
from constants import EPOCHS_PER_GENERATION, LENGTH_EXP, MOVE_THRESHOLD, ACTION_THRESHOLD
import numpy as np
from scipy.special import expit
from structs import Direction, Coordinate


# TODO: Refactor every lambda, only input is creature
class Neuron(Enum):
    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name},{self.value}:\t{self.description}>"


class InputNeuron(Neuron):
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        obj.label = args[1]
        obj.description = args[2]
        obj.function = args[3]
        return obj

    slr = (1, 'slr', 'pheromone_gradient_left_right',
           lambda **kwargs:
           kwargs['world'].pheromone_on_direction(kwargs['creature'].position,
                                                  kwargs['creature'].direction.rotate(2)))  # pheromone gradient l/r
    sfd = (2, 'sfd', 'pheromone_gradient_forward',
           lambda **kwargs:
           kwargs['world'].pheromone_on_direction(kwargs['creature'].position,
                                                  kwargs['creature'].direction))  # pheromone gradient forward
    sg = (3, 'sg', 'pheromone_density',
          lambda **kwargs:
          kwargs['world'].pheromone_density(kwargs['creature'].position))  # pheromone density
    age = (4, 'age', 'instance_age',
           lambda **kwargs:
           kwargs['creature'].age / EPOCHS_PER_GENERATION)  # instance age
    rdn = (5, 'rdn', 'random_input',
           lambda **kwargs: np.random.rand())  # random input
    blr = (6, 'blr', 'blockage_left_right',
           lambda **kwargs:
           kwargs['world'].blocks_on_direction(kwargs['creature'].position,
                                               kwargs['creature'].direction.rotate(2)))  # blockage left right
    osc = (7, 'osc', 'oscillator',
           lambda **kwargs:
           (np.sin(np.pi * kwargs['world'].epoch / (kwargs['creature'].oscillator_period / 2)) + 1) / 2)  # oscillator
    bfd = (8, 'bfd', 'bfd_blockage_forward',
           lambda **kwargs:
           kwargs['world'].blocks_on_direction(kwargs['creature'].position,
                                               kwargs['creature'].direction))  # bfd blockage forward
    plr = (9, 'plr', 'population_gradient_left_right',
           lambda **kwargs:
           kwargs['world'].population_on_direction(kwargs['creature'].position,
                                                   kwargs['creature'].direction.rotate(2)))  # population gradient l/r
    pop = (10, 'pop', 'population_density',
           lambda **kwargs:
           kwargs['world'].population_density(kwargs['creature'].position))  # population density
    pfd = (11, 'pfd', 'population_gradient_forward',
           lambda **kwargs:
           kwargs['world'].population_on_direction(kwargs['creature'].position,
                                                   kwargs['creature'].direction))  # population gradient forward
    lpf = (12, 'lpf', 'population_long_range_forward',
           lambda **kwargs:
           kwargs['world'].population_next_creature_on_direction(kwargs['creature'].position,
                                                                 kwargs['creature'].direction,
                                                                 kwargs[
                                                                     'creature'].long_probe_distance))  # next creature
    lmy = (13, 'lmy', 'last_movement_Y',
           lambda **kwargs:
           kwargs['creature'].previous_position.y * kwargs['world'].size.y)  # last movement Y
    lbf = (14, 'lbf', 'blockage_long_range_forward',
           lambda **kwargs:
           kwargs['world'].block_next_on_direction(kwargs['creature'].position,
                                                   kwargs['creature'].direction,
                                                   kwargs['creature'].long_probe_distance
                                                   ))  # blockage long range forward
    lmx = (15, 'lmx', 'last_movement_X',
           lambda **kwargs:
           kwargs['creature'].previous_position.x * kwargs['world'].size.x)  # last movement X
    bdy = (16, 'bdy', 'north_south_border_distance',
           lambda **kwargs:
           kwargs['world'].distance_to_border(kwargs['creature'].position,
                                              Direction.N))  # north south border distance
    gen = (17, 'gen', 'genetic_similarity_of_fwd_neighbor',
           lambda **kwargs:
           kwargs['creature'].genetic_similarity_with_neighbor())  # gen sim of fwd neighbor

    bdx = (18, 'bdx', 'east_west_border_distance',
           lambda **kwargs:
           kwargs['world'].distance_to_border(kwargs['creature'].position,
                                              Direction.E))  # east west border distance
    lx = (19, 'lx', 'east_west_world_location',
          lambda **kwargs:
          kwargs['creature'].position.x / kwargs['world'].size.x)  # east west world location
    bd = (20, 'bd', 'nearest_border_distance',
          lambda **kwargs:
          kwargs['world'].nearest_border_distance(kwargs['creature'].position))  # nearest border distance
    ly = (21, 'ly', 'north_south_world_location',
          lambda **kwargs:
          kwargs['creature'].position.y / kwargs['world'].size.y)  # north south world location

    pheromone_gradient_left_right = slr
    pheromone_gradient_forward = sfd
    pheromone_density = sg
    instance_age = age
    random_input = rdn
    blockage_left_right = blr
    oscillator = osc
    bfd_blockage_forward = bfd
    population_gradient_left_right = plr
    population_density = pop
    population_gradient_forward = pfd
    population_long_range_forward = lpf
    last_movement_Y = lmy
    blockage_long_range_forward = lbf
    last_movement_X = lmx
    north_south_border_distance = bdy
    genetic_similarity_of_fwd_neighbor = gen
    east_west_border_distance = bdx
    east_west_world_location = lx
    nearest_border_distance = bd
    north_south_world_location = ly


class InnerNeuron(Neuron):
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        obj.label = args[1]
        obj.description = args[2]
        obj.function = args[3]
        return obj

    sigmoid = (1, 'sig', 'sigmoid', expit)
    relu = (2, 'relu', 'ReLU', lambda x: max(0, x))
    l_relu = (3, 'leaky_relu', 'Leaky_ReLU', lambda x: max(0.01 * x, x))
    tanh = (4, 'tanh', 'Hyperbolic tangent', lambda x: np.tanh(x))

    ReLU = relu
    Leaky_ReLU = l_relu


class ActionNeuron(Neuron):
    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        obj.label = args[1]
        obj.description = args[2]
        obj.function = args[3]
        return obj

    lpd = (1, 'lpd', 'set_long_probe_distance',
           lambda x, creature: ('long_probe_distance', creature['long_probe_distance'] if x <= ACTION_THRESHOLD
           else (1 + np.tanh(x)) ** LENGTH_EXP))
    osc = (2, 'osc', 'set_oscillator_period',
           lambda x, creature: ('oscillator_period', creature['oscillator_period'] if x <= ACTION_THRESHOLD
           else (1 + np.tanh(x)) ** LENGTH_EXP))
    res = (3, 'res', 'set_responsiveness',
           lambda x, creature: ('responsiveness', creature['responsiveness'] if x <= ACTION_THRESHOLD
           else (1 + np.tanh(x)) ** LENGTH_EXP))
    mrn = (4, 'mrn', 'move_random',
           lambda x, creature: ('move', Direction.get_random_direction().get_coordinate()))
    mrl = (5, 'mrl', 'move_left_right',
           lambda x, creature:
           ('move', Coordinate(0, 0) if x <= ACTION_THRESHOLD
           else creature.direction.rotate(2).get_coordinate() if x > MOVE_THRESHOLD
           else creature.direction.rotate(-2).get_coordinate()))
    my = (6, 'my', 'move_north_south',
          lambda x, creature:
          ('move', Coordinate(0, 0) if x <= ACTION_THRESHOLD
          else Direction['N' if x > MOVE_THRESHOLD else 'S'].get_coordinate()))
    sg = (7, 'sg', 'emit_pheromone',
          lambda x, creature: ('emit_pheromone', max(0, np.tanh(x))))
    mfd = (8, 'mfd', 'move_forward',
           lambda x, creature: ('move', Coordinate(0, 0) if x <= ACTION_THRESHOLD
           else creature.direction.get_coordinate()))
    mrv = (9, 'mrv', 'move_backward',
           lambda x, creature: ('move', Coordinate(0, 0) if x <= ACTION_THRESHOLD
           else creature.direction.rotate(4).get_coordinate()))
    mx = (10, 'mx', 'move_east_west',
          lambda x, creature: ('move', Coordinate(0, 0) if x <= ACTION_THRESHOLD
          else Direction['E' if x > MOVE_THRESHOLD else 'S'].get_coordinate()))
    kill = (11, 'kill', 'kill_forward_neighbor',
            lambda x, creature: ('kill', creature.position + creature.direction.get_coordinate()))

    set_long_probe_distance = lpd
    set_oscillator_period = osc
    set_responsiveness = res  # Makes the creature more nervous
    move_random = mrn
    move_left_right = mrl
    move_north_south = my
    kill_forward_neighbor = kill
    emit_pheromone = sg
    move_forward = mfd
    move_backward = mrv
    move_east_west = mx
