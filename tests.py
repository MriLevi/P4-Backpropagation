import activation as act
import neuron_network as nn
import itertools
import unittest
from collections import defaultdict

'''2.aIk heb hieronder de tests staan. 4 tests falen, dat zijn de tests zoals ik ze voor de perceptron ook heb gebruikt.
 Deze tests falen omdat er uit sigmoid eigenlijk nooit 0.0 komt, En bijna nooit 1.0. Ik heb ook tests aangemaakt waar
 ik de outputs van de gates afrond - dan werken ze wel weer.
 Dit verklaard dus ook waarom 4 van de 5 tests falen. 
 Verder heb ik helemaal onderaan de test staan voor een halfadder. Ook hier ronden we de outputs af.'''

class Tests(unittest.TestCase):
    def test_and_gate(self):
        """the and gate outputs 1 if both x1 and x2 (inputs) are 1, else it outputs 0"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [1, 1]
        threshold = 1
        and_neuron = nn.Neuron(threshold, act.sigmoid, weights)

        for i in inputs:
            if i == (1, 1):
                self.assertEqual(1, and_neuron.determine_output(i),
                                 f"UNROUNDED AND gate: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, and_neuron.determine_output(i),
                                 f"UNROUNDED AND gate: Result should be 0, since input is {i}")

    def test_rounded_and_gate(self):
        """De AND gate werkt niet als we de output niet afronden. Deze versie rond de output af en dan werkt hij wel."""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [1, 1]
        threshold = 1
        and_neuron = nn.Neuron(threshold, act.sigmoid, weights)

        for i in inputs:
            if i == (1, 1):
                self.assertEqual(1, round(and_neuron.determine_output(i)),
                                 f"AND gate: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, round(and_neuron.determine_output(i)),
                                 f"AND gate: Result should be 0, since input is {i}")
    def test_or_gate(self):
        """The or gate returns 1 if either x1 or x2 is 1, or if both are 1."""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [2, 2]
        threshold = 1
        or_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0):
                self.assertEqual(0, or_neuron.determine_output(i),
                                 f"OR gate: Result should be 0, since input is {i}")
            else:
                self.assertEqual(1, or_neuron.determine_output(i),
                                 f"OR gate: Result should be 1, since input is {i}")
    def test_rounded_or_gate(self):
        """Ook de OR-gate werkt alleen als we de values afronden"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [2, 2]
        threshold = 1
        or_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0):
                self.assertEqual(0, round(or_neuron.determine_output(i)),
                                 f"ROUNDED OR gate: Result should be 0, since input is {i}")
            else:
                self.assertEqual(1, round(or_neuron.determine_output(i)),
                                 f"ROUNDED OR gate: Result should be 1, since input is {i}")
    def test_not_gate(self):
        """The NOT gate returns 1 if inputs are x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [-1, -1]
        threshold = -1
        not_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0):
                self.assertEqual(1, not_neuron.determine_output(i),
                                 f"NOT gate: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, not_neuron.determine_output(i),
                                 f"NOT gate: Result should be 0, since input is {i}")
    def test_rounded_not_gate(self):
        """The NOT gate returns 1 if inputs are x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [-1, -1]
        threshold = -1
        not_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0):
                self.assertEqual(1, round(not_neuron.determine_output(i)),
                                 f"ROUNDED NOT gate: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, round(not_neuron.determine_output(i)),
                                 f"ROUNDED NOT gate: Result should be 0, since input is {i}")
    def test_nor_gate(self):
        """The NOR gate returns 1 only if x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=3))
        weights = [-1, -1, -1]
        threshold = -1  # bias == -threshold
        nor_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0, 0):
                self.assertEqual(1, nor_neuron.determine_output(i), f"UNROUNDED NOR: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, nor_neuron.determine_output(i), f"UNROUNDED NOR: Result should be 1, since input is {i}")

    def test_rounded_nor_gate(self):
        """The NOR gate returns 1 only if x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=3))
        weights = [-1, -1, -1]
        threshold = -1  # bias == -threshold
        nor_neuron = nn.Neuron(threshold, act.sigmoid, weights)
        for i in inputs:
            if i == (0, 0, 0):
                self.assertEqual(1, round(nor_neuron.determine_output(i)), f"ROUNDED NOR: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, round(nor_neuron.determine_output(i)), f"ROUNDED NOR: Result should be 1, since input is {i}")

    def test_half_adder(self):
        inputs = list(itertools.product([0,1], repeat=2))
        #initialize the gates for layer 1
        or_neuron = nn.Neuron(1, act.sigmoid, [2,2])
        nand_neuron = nn.Neuron(-1.5, act.sigmoid, [-1,-1])
        and_neuron = nn.Neuron(1.5, act.sigmoid, [1,1])
        layer1 = nn.NeuronLayer([or_neuron, nand_neuron, and_neuron])

        #layer2, output layer
        sum_neuron = nn.Neuron(1.5, act.sigmoid, [1.5,1.5,-1])
        carry_neuron = nn.Neuron(1, act.sigmoid, [-0.5,-0.5,3])
        layer2 = nn.NeuronLayer([sum_neuron, carry_neuron])

        half_adder = nn.NeuronNetwork([layer1, layer2])

        # half adder should return [0,0] for input [0,0], [1,0] for inputs [1,0] [0,1], and [0,1] for input [1,1].
        for i in inputs:
            if i == (0,0):
                self.assertEqual([0,0], [round(i) for i in half_adder.feed_forward(i)])
            elif i in [(0,1),(1,0)]:
                self.assertEqual([1,0], [round(i) for i in half_adder.feed_forward(i)])
            else:
                self.assertEqual([0,1], [round(i) for i in half_adder.feed_forward(i)])


if __name__ == '__main__':
    unittest.main()
