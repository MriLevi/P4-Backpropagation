import activation as act
import neuron_network as nn
import itertools
import unittest
import random as rand


class MyTestCase(unittest.TestCase):
    def test_AND(self):
        inputs = list(itertools.product([0, 1], repeat=2))
        targets = [[0],[0],[0],[1]]
        weights = [rand.random(), rand.random()]
        threshold = rand.random()
        epochs=1500
        and_neuron = nn.Neuron(threshold, act.sigmoid, weights, learning_rate=0.1)
        and_layer = nn.NeuronLayer([and_neuron])
        and_network = nn.NeuronNetwork([and_layer])

        and_network.train(inputs, targets, epochs)

        for i in inputs:
            if i == (1, 1):
                self.assertEqual(1, round(and_neuron.determine_output(i)),
                                 f"AND gate: Result should be 1, since input is {i}")
            else:
                self.assertEqual(0, round(and_neuron.determine_output(i)),
                                 f"AND gate: Result should be 0, since input is {i}")

    def test_XOR(self):
        learning_rate = 0.5
        #layer1
        neuron1_1 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random()], learning_rate)
        neuron1_2 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random()], learning_rate)
        layer1 = nn.NeuronLayer([neuron1_1, neuron1_2])

        #layer2
        neuron2_1 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random()], learning_rate)
        layer2 = nn.NeuronLayer([neuron2_1])
        XOR_network = nn.NeuronNetwork([layer1, layer2])

        inputs = list(itertools.product([0, 1], repeat=2))
        targets = [[0], [1], [1], [0]]
        epochs = 10000

        XOR_network.train(inputs, targets, epochs)

        for i in inputs:
            XOR_network.feed_forward(i)
            if i in [(0,1), (1,0)]:
                self.assertEqual(1, round(XOR_network.all_outputs[-1][0]))
            else:
                self.assertEqual(0, round(XOR_network.all_outputs[-1][0]))




    def test_Half_adder(self):
        inputs = list(itertools.product([0, 1], repeat=2))
        targets = [[0,0], [1,0], [1,0], [0,1]]
        learning_rate = 1
        epochs = 10000
        # initialize the gates for layer 1
        neuron1_1 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random()], learning_rate)
        neuron1_2 = nn.Neuron(rand.random(), act.sigmoid, [1,1], learning_rate)
        neuron1_3 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random()], learning_rate)
        layer1 = nn.NeuronLayer([neuron1_1, neuron1_2, neuron1_3])

        # layer2, output layer
        neuron2_1 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random(), rand.random()], learning_rate)
        neuron2_2 = nn.Neuron(rand.random(), act.sigmoid, [rand.random(),rand.random(), rand.random()], learning_rate)
        layer2 = nn.NeuronLayer([neuron2_1, neuron2_2])

        half_adder = nn.NeuronNetwork([layer1, layer2])
        half_adder.train(inputs, targets, epochs)

        # half adder should return [0,0] for input [0,0], [1,0] for inputs [1,0] [0,1], and [0,1] for input [1,1].
        for i in inputs:
            half_adder.feed_forward(i)
            if i == (0, 0):
                self.assertEqual([0, 0], [round(i) for i in half_adder.all_outputs[-1]])
            elif i in [(0, 1), (1, 0)]:
                self.assertEqual([1, 0], [round(i) for i in half_adder.all_outputs[-1]])
            else:
                self.assertEqual([0, 1], [round(i) for i in half_adder.all_outputs[-1]])



if __name__ == '__main__':
    unittest.main()