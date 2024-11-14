import numpy as np


# Perceptron for OR operation
class Perceptron_OR:
    def __init__(self, input_size=2):
        self.weights = [1, 1]
        self.bias = -1
        print("OR weights:", self.weights)
        print("OR bias:", self.bias)

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(sum)


# Perceptron for AND operation
class Perceptron_ET:
    def __init__(self, input_size=2):
        self.weights = [0.5, 0.5]
        self.bias = -1
        print("AND weights:", self.weights)
        print("AND bias:", self.bias)

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(sum)


# XOR Perceptron using OR, AND, and NOT logic
class Perceptron_XOR:
    def __init__(self):
        self.perceptron_or = Perceptron_OR()
        self.perceptron_and = Perceptron_ET()

    def predict(self, inputs):
        # Compute the OR and AND values
        or_output = self.perceptron_or.predict(inputs)
        and_output = self.perceptron_and.predict(inputs)

        # NOT operation (invert AND output)
        not_and_output = 1 if and_output == 0 else 0

        # Final XOR output using AND between OR and NOT(AND)
        xor_output = self.perceptron_and.predict([or_output, not_and_output])
        return xor_output


# Testing the Perceptron_XOR with all possible inputs
xor_perceptron = Perceptron_XOR()
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("\nXOR Perceptron Results:")
for inputs in test_inputs:
    print(f"XOR({inputs[0]}, {inputs[1]}) = {xor_perceptron.predict(inputs)}")
