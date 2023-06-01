#!/usr/bin/env python
# coding: utf-8

# What is the function of a summation junction of a neuron? What is threshold activation function?
# 

# The summation junction of a neuron computes the weighted sum of its inputs, at the same time as the edge activation function determines if the neuron need to set off based totally on a threshold. In Python, you can put in force these as features: summation_junction calculates the entire enter signal, and threshold_activation returns the activation output based at the enter and threshold.

# What is a step function? What is the difference of step function with threshold function?
# 

# A step characteristic is a mathematical characteristic that has a constant output for any input value above a sure threshold and a exclusive steady output for any input fee underneath the brink.
# 
# def step_functon(x, threshold):
#     return 1 if x >= threshold else 0
# 
# 
# The difference among a step function and a threshold function is that a step feature has a steady output above and underneath the brink, even as a threshold characteristic typically has wonderful output values on either side of the brink. The step function keeps its output as soon as the enter crosses the brink, whereas a threshold function may also exchange its output value as the input varies round the brink.

# Explain the McCulloch–Pitts model of neuron.
# 

# The McCulloch- pitts model of a neuron , proposed by warren McCulloch And walter pitts in 1943 is one of the earliest mathematical model of a biological neuron. it provides a simplified representation of how a neuron processes input nad produces as output.
# 
# The neuron take binary inputs and assigns weight to each input.
# it sums up the weighted inputs.
# if the sum exceeds or equals a certain threshold, the neuron fires and output1
# otherwise it remain in active and output 0

# Explain the ADALINE network model.
# 

# The ADALINE (Adaptive Linear Neuron) network model is a type of neural network that uses a linear activation function and the delta rule for weight adjustment. 
# 
# THe ADALINE model consists of a single layer of neurons.
# it take multiple inputs and assigns weight to each input.
# the weighted sum of input is calculated
# the linear activition function , which is the identity function is applied to the weighted sum 
# the output is equal to the weighted sum of input.
# the delta rule is used to adjust the weights during traning.
# 

# What is the constraint of a simple perceptron? Why it may fail with a real-world data set?
# 

# a perceptron model has limitation as follow:
#     the output of a perceptron can only by a binary number (0 or 1)due to the hard limit transfer function . preceptron can only be used to classify the linearly separable sets of input vector .if input vectors are non-linear, it is not easy to classify them properly.

# What is linearly inseparable problem? What is the role of the hidden layer?
# 

# the linearly inseparable problem occurs when data point from different classes cannot be seperated by a strainght line or hyperplane.hidden layer in a neuron network play a crucial role in addressing this problem by allowing the network to learn and represent non linear realtionshhip in the data.

# Explain XOR problem in case of a simple perceptron.
# 

# the XOR is a classic example that demostrates the limitatioin of a simple perceptron as it fails to accurately classify XOR is a logical operation where the output is true(1) only of the input is true (1). a simple perceptron cannot learn the XOR function because it requires non linear decision boundaries. to solve the XOR problems more advances neural network architectures such As multi layer perceptron (MPLs) are needed to capture non linear relationship and achieve accurate predictions.

# Design a multi-layer perceptron to implement A XOR B.
# 

# In[10]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1

weights_hidden = np.random.randn(input_size, hidden_size)
biases_hidden = np.random.randn(hidden_size)
weights_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    hidden_layer_input = np.dot(inputs, weights_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    
    output_error = targets - output_layer_output
    output_delta = output_error * output_layer_output * (1 - output_layer_output)
    hidden_error = np.dot(output_delta, weights_output.T)
    hidden_delta = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
    
    weights_output += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0) * learning_rate
    weights_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
    biases_hidden += np.sum(hidden_delta, axis=0) * learning_rate

hidden_layer_output = sigmoid(np.dot(inputs, weights_hidden) + biases_hidden)
output_layer_output = sigmoid(np.dot(hidden_layer_output, weights_output) + bias_output)

for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Target: {targets[i]}, Predicted: {output_layer_output[i]}")


# Explain the single-layer feed forward architecture of ANN.
# 

# In[11]:


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

weights = np.array([[0.5, 0.5]])

hidden_layer_output = sigmoid(np.dot(inputs, weights.T))

for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Output: {hidden_layer_output[i]}")


# Explain the competitive network architecture of ANN.
# 

# In[12]:


class CompetitiveNetwork:
    def __init__(self, input_size, num_neurons):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_neurons, input_size)

    def activate(self, input):
        activations = np.dot(self.weights, input)
        winner_index = np.argmax(activations)
        return winner_index

    def train(self, inputs, epochs):
        for epoch in range(epochs):
            for input in inputs:
                winner_index = self.activate(input)
                self.weights[winner_index] += input

network = CompetitiveNetwork(input_size=2, num_neurons=2)

network.train(inputs, epochs=100)

for input in inputs:
    winner_index = network.activate(input)
    print(f"Input: {input}, Winner Index: {winner_index}")


# Consider a multi-layer feed forward neural network. Enumerate and explain steps in the backpropagation algorithm used to train the network.
# 

# Forward Pass: The input data is propagated through the network layer by layer, computing the activations of each neuron using weighted sums and activation functions. This process moves from the input layer to the output layer, generating predictions.
# 
# Compute Loss: The predicted output is compared to the desired output, and a loss function is used to quantify the difference between them. Common loss functions include mean squared error (MSE) for regression problems or cross-entropy for classification problems.
# 
# Backward Pass: Starting from the output layer, the gradient of the loss with respect to each weight and bias in the network is calculated. This is achieved by applying the chain rule of calculus to propagate the error back through the layers. The gradients quantify how each weight and bias contributes to the overall error.
# 
# Update Weights and Biases: The calculated gradients are used to update the weights and biases of the network. This update is performed using an optimization algorithm, such as stochastic gradient descent (SGD), which adjusts the weights and biases by subtracting a portion of the gradient multiplied by a learning rate. This process iteratively improves the network's performance by minimizing the loss.
# 
# Iterate: Steps 1 to 4 are repeated for a specified number of iterations or until a convergence criterion is met. Typically, the training data is divided into smaller batches, and the backpropagation steps are applied iteratively on each batch. This approach, called mini-batch stochastic gradient descent, helps accelerate the training process and promotes better generalization.

# What are the advantages and disadvantages of neural networks?
# 

# advantages of neural network:
#     Neural networks can learn complex patterns and recognize intricate relationships in data.
# They can model non-linear relationships effectively.
# Neural networks can process data in parallel, making them efficient for certain tasks.
# They can generalize well to unseen data if trained on diverse and representative examples.
# 
# Disadvantages of Neural Networks:
# 
# Neural networks can be computationally expensive and require significant resources.
# They are often seen as "black boxes" with limited interpretability.
# Neural networks need a large amount of training data to learn effectively.
# Choosing optimal hyperparameters can be challenging.
# There is a risk of overfitting, where the model performs well on training data but fails to generalize to new data.
# 
# 
# 
# 

# Biological neuron
# ReLU function
# 

# Biological Neurons:
# 
# Biological neurons are the basic units of the human brain and nervous system.
# They receive and process signals through dendrites, integrate them in the cell body, and transmit them through the axon.
# When the integrated signal exceeds a certain threshold, the neuron fires an electrical impulse called an action potential.
# ReLU Activation Function:
# 
# ReLU (Rectified Linear Unit) is a popular activation function in neural networks.
# It outputs the input value if it's positive, and zero otherwise.
# ReLU introduces non-linearity, enabling neural networks to learn complex patterns.
# It helps address the vanishing gradient problem and promotes efficient gradient propagation.
# ReLU is computationally efficient and supports better generalization.
# However, some neurons can become inactive in the "dying ReLU" problem, leading to limited learning.

# In[ ]:




