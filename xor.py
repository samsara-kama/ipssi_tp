import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, nb_epochs=100, learning_rate=0.1):
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)  # includes bias weight
        print("Initial weights:", self.weights)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.nb_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction) * 1  # bias adjustment
                print("Updated weights:", self.weights)

# Training data for the XOR function
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected XOR outputs
expected_outputs_xor = np.array([0, 1, 1, 0])

# Step 1: Train the OR perceptron
print("\nTraining OR Perceptron:")
perceptron_or = Perceptron(2)
expected_outputs_or = np.array([0, 1, 1, 1])
perceptron_or.train(training_inputs, expected_outputs_or)

# Step 2: Train the AND perceptron
print("\nTraining AND Perceptron:")
perceptron_and = Perceptron(2)
expected_outputs_and = np.array([0, 0, 0, 1])
perceptron_and.train(training_inputs, expected_outputs_and)

# Step 3: Train the XOR perceptron, using OR and AND outputs as inputs
print("\nTraining XOR Output Perceptron:")
perceptron_xor_output = Perceptron(2)
layer1_outputs = []
for inputs in training_inputs:
    or_output = perceptron_or.predict(inputs)
    and_output = perceptron_and.predict(inputs)
    layer1_outputs.append([or_output, and_output])
layer1_outputs = np.array(layer1_outputs)

# Train the output XOR neuron using the layer1 outputs as its input
perceptron_xor_output.train(layer1_outputs, expected_outputs_xor)

# Testing the XOR Perceptron Network
print("\nTesting XOR Perceptron Network:")
for inputs in training_inputs:
    or_output = perceptron_or.predict(inputs)
    and_output = perceptron_and.predict(inputs)
    xor_output = perceptron_xor_output.predict([or_output, and_output])
    print(f"XOR({inputs[0]}, {inputs[1]}) = {xor_output}")
