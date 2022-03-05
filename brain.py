from collections import defaultdict
from functions import value_from_gene
from neuron import Neuron, InnerNeuron, InputNeuron, ActionNeuron

# Brain class
from structs import Coordinate


class Brain:
    def __init__(self, genome, creature):

        self.connections = {
            'input': defaultdict(list),
            'inner': defaultdict(list),
            'action': []
        }
        self.creature = creature
        self._neuron_values: dict[Neuron: float] = self._create_neurons(genome)

        self._clean_connections()

    def _create_neurons(self, genome) -> dict:
        values = {}
        for gene in genome:
            # Identifies the source based on first bit of gene
            if value_from_gene(gene, 'source_type'):  # Merge in one function, they do the same
                source = InnerNeuron(value_from_gene(gene, 'source_id') % len(InnerNeuron) + 1)
                source_name = 'inner'
            else:
                source = InputNeuron(value_from_gene(gene, 'source_id') % len(InputNeuron) + 1)
                source_name = 'input'

            if value_from_gene(gene, 'dest_type'):
                dest = InnerNeuron(value_from_gene(gene, 'dest_type') % len(InnerNeuron) + 1)
            else:
                dest = ActionNeuron(value_from_gene(gene, 'dest_id') % len(ActionNeuron) + 1)
                self.connections['action'].append(dest)

            values[source] = 0.0
            values[dest] = 0.0  # Naive but will do

            weight = (value_from_gene(gene, 'weight') - 32768) / 8000

            self.connections[source_name][source].append((dest, weight))

            # TODO: Check unenviable connections to save processing power

        return values

    def _clean_connections(self):
        for key in ('input', 'inner'):
            for origin, dest in self.connections[key].items():
                new_dest = [item for item in dest if (not isinstance(item[0], InnerNeuron) or
                                                      item[0] in self.connections['inner'])]
                if len(new_dest) != len(dest):
                    # TODO: pop empty key
                    self.connections[key][origin] = new_dest

    #  This should be on creature (?)
    def calculate_step(self):
        next_move = Coordinate(0, 0)

        # First calculate input neurons
        for origin, dest_arr in self.connections['input'].items():
            # If world is in creature, maybe just send creature
            value = origin.function(world=self.creature.world, creature=self.creature)
            for dest in dest_arr:
                self._neuron_values[dest[0]] += (dest[1] * value)

        for origin, dest_arr in self.connections['inner'].items():
            value = origin.function(self._neuron_values[origin])
            for dest in dest_arr:
                self._neuron_values[dest[0]] += dest[1] * value

        for origin in set(self.connections['action']):
            action_result = origin.function(self._neuron_values[origin], self.creature)
            match action_result[0]:
                case 'move':
                    next_move += action_result[1]
                case 'emit_pheromone':
                    self.creature.emit_pheromone(action_result[1])
                case 'kill':
                    continue  # I'm pacifist
                case _:
                    self.creature[action_result[0]] = action_result[1]

        return next_move
