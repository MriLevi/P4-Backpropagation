class Neuron:

    def __init__(self, threshold, activation_function, weights, learning_rate=0.01):

        self.bias = -threshold
        self.activation_function = activation_function
        self.weights = weights
        self.learning_rate = learning_rate
        self.inputs = []
        self.output = None

        self.error = 0
        self.gradients = None
        self.target = None
        self.derivative = None
        self.delta_weights = None
        self.delta_bias = None

    def __str__(self):
        return f'bias: {self.bias} - activation_function: {self.activation_function.__name__}\n' \
               f'weights: {self.weights}'

    def dot(self, x1, x2):
        """returns the inner product of two vectors"""
        """NOTE: Some checking should be done to verify the dimensions of the input - this is lazy"""
        return sum([x * y for x, y in zip(x1, x2)])

    def determine_output(self, inputs):
        """Determines the output of a neuron by adding the dot product of the weights and inputs and adding bias"""
        self.inputs = inputs
        self.output = self.activation_function(self.bias + self.dot(self.weights, inputs))
        return self.output

    def hidden_error(self, weights, errors):
        """Determine the error for hidden neuron"""
        self.error = self.derivative * self.dot(weights, errors)

    def output_error(self, target):
        """Determine the  error for output neuron"""
        self.target = target
        self.error = self.derivative * -(target - self.output)

    def update(self):
        """Adjust the weights and bias of the neuron"""
        new_weights = [i - j for i, j in zip(self.weights, self.delta_weights)]
        self.weights = new_weights
        self.bias -= self.delta_bias

    def neuron_function_activation(self, weights=None, errors=None, target=None):
        """This neuron activation function runs the necessary functions.
        We determine derivative, gradients, new weights and new biases in this function.
        We also determine either hidden or output errors.
        If a target is specified, we use the output_error function as that indicates that the neuron is an output neuron.
        Else, we use the hidden error function."""

        self.derivative = self.output * (1 - self.output)

        if target is not None:
            self.target = target
            self.output_error(target)
        else:
            self.hidden_error(weights, errors)
        self.gradients = [i * self.error for i in self.inputs]
        self.delta_weights = [self.learning_rate * i for i in self.gradients]
        self.delta_bias = self.learning_rate * self.error


class NeuronLayer:

    def __init__(self, neurons):
        self.neurons = neurons

    def feed_forward(self, inputs):
        """Determine the outputs for neurons in this layer"""
        outputs = [neuron.determine_output(inputs) for neuron in self.neurons]
        return outputs

    def layer_errors(self):
        """Calculate all the errors in this layer"""
        return [neuron.error for neuron in self.neurons]

    def layer_update(self):
        """Update all weigths and biases in this layer by calling neuron.update()"""
        for neuron in self.neurons:
            neuron.update()


class NeuronNetwork:

    def __init__(self, neuron_layers):
        self.neuron_layers = neuron_layers
        self.all_outputs = []
        self.network_losses = []

    def feed_forward(self, initial_input):
        """Feed forward determines the output for each layer by using the output of the previous as input for the next
            It determines the first layers outputs outside of the loop to be able to then loop over the remaining
            layers. Finally, it returns the output of the final layer.
            NOTE: The first layer must always have its inputs defined."""
        inputs = self.neuron_layers[0].feed_forward(initial_input)  # input layer
        for neuron_layer in self.neuron_layers[1:]:
            inputs = neuron_layer.feed_forward(inputs)
        self.all_outputs = [initial_input]
        self.all_outputs.append(inputs)

    def loss(self, targets):
        """Calculate the loss for a network"""
        loss = 0
        for i in range(len(targets)):
            loss += (targets[i] - self.all_outputs[-1][i]) ** 2
        self.network_losses.append(loss)

    def total_loss(self, inputs, targets):
        """This function determines the total loss"""
        for input, target in zip(inputs, targets):
            self.feed_forward(input)
            self.loss(target)
        return sum(self.network_losses) / len(self.network_losses)

    def update(self):
        """Update every layers weights and biases"""
        for layer in self.neuron_layers:
            layer.layer_update()

    def train(self, inputs, targets, epochs):
        """This function trains the neural network"""

        count = 0
        loss_target = 0.01
        while count < epochs and self.total_loss(inputs, targets) > loss_target:
            for input, target in zip(inputs, targets):
                self.feed_forward(input)

                output_layer = self.neuron_layers[-1].neurons
                hidden_layer = self.neuron_layers[:-1]

                # output layer
                for i, neuron in enumerate(output_layer):
                    neuron.neuron_function_activation(weights=None, errors=None, target=target[i])

                # calculate hidden neurons
                for layer in reversed(range(len(hidden_layer))):
                    for index in range(len(hidden_layer[layer].neurons)):
                        weights = [neuron.weights[index] for neuron in self.neuron_layers[layer + 1].neurons]
                        errors = self.neuron_layers[layer + 1].layer_errors()
                        # activate the neuron
                        self.neuron_layers[layer].neurons[index].neuron_function_activation(weights, errors)

                for layer in self.neuron_layers:
                    layer.layer_update()
                self.network_losses = [] #reset the network losses
            count += 1 #keep track of epochs
