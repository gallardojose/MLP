import re
import math
import random
import numpy as np

def read_file(filename):
    # feature_vecs = [ [id, vector, label], ... ]
    #example feature vecs = [ [2, [91,34,45,34], 5], [7, [45,33,78,12], 9], ... ]

    feature_vecs = [];
    file = open(filename, "r")

    for line in file:
        vectors = []
        data = re.split(r'[()\s]\s*', line)
        while '' in data:
            data.remove('')

        for item in data[1:-1]:
            vectors.append(int(item))

        feature_vecs.append([int(data[0]), vectors , int(data[-1])])

    return feature_vecs

def normal(vec, max):
    res = []
    for val in vec:
        res.append(val / max)
    return res

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def swixi(w, x):
    sum = 0
    for i in range(len(w)):
        sum = sum + (w[i] * x[i])
    return sum

def init_weights(rol, col):
    return np.random.randn(rol, col) * np.sqrt(1.0 / col)

def forward_propagation(training_vec):
    # input layer to hidden layer
    hidden_layer = []
    for i in range(num_hidden_nodes):
        hidden_layer.append(logistic(swixi(W1[i], training_vec)))

    # hidden layer to ouput layer
    output_layer = []
    for i in range(num_output_nodes):
        output_layer.append(logistic(swixi(W2[i], hidden_layer)))

    return hidden_layer, output_layer

def back_propagation(learning_rate, iteration):
    for it in range(iteration):
        if it % 100 == 0:
            print("iteration", it)

        for value in training_set:
            training_vec = normal(value[1], max_feature_value)
            hidden_layer, output_layer = forward_propagation(training_vec)

            # Calculate errors
            delta1 = [] # output delta
            for j in range(num_output_nodes):
                y = output_layer[j]
                t = 0.2
                if j == value[2]:
                    t = 0.8
                delta1.append(y * (1 - y) * (t - y))

            delta2 = [] # hidden layer delta
            for j in range(num_hidden_nodes):
                h = hidden_layer[j]
                s = swixi(W2[:,j], delta1)
                delta2.append(h * (1 - h) * s)

            # Update Weights
            for j in range(num_hidden_nodes):
                for i in range(num_output_nodes):
                    W2[i][j] = W2[i][j] + learning_rate * delta1[i] * hidden_layer[j]
                for k in range(num_input_nodes):
                    W1[j][k] = W1[j][k] + learning_rate * delta2[j] * training_vec[k]



classified_set = read_file("ClassifiedSetData.txt")
training_set = classified_set

# number of hidden nodes = number of features
num_input_nodes = len(training_set[0][1])
num_hidden_nodes = num_input_nodes
num_output_nodes = 8
max_feature_value = 96.0

W1 = init_weights(num_hidden_nodes, num_input_nodes) # weight matrix from input to hidden layer
W2 = init_weights(num_output_nodes, num_hidden_nodes) # weight matrix from hidden to output layer
back_propagation(0.1, 5000)

for data in training_set:
    hl, ol = forward_propagation(normal(data[1]), max_feature_value)
    print(data[0], ol)
