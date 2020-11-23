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

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def swixi(w, x):
    sum = 0
    for i in range(len(w)):
        sum = sum + (w[i] * x[i])
    return sum

def init_weights(size):
    return np.random.randn(size) * np.sqrt(1.0 / size)

def forward_propagation():
    for value in training_set:
        # input layer to hidden layer
        hidden_layer = []
        for i in range(num_hidden_nodes):
            w = init_weights(num_input_nodes)
            x = value[1]
            hidden_layer.append(logistic(swixi(w, x)))

        # hidden layer to ouput layer
        output_layer = []
        for i in range(num_output_nodes):
            w = init_weights(num_hidden_nodes)
            x = hidden_layer
            output_layer.append(logistic(swixi(w, x)))
        print(output_layer)


classified_set = read_file("ClassifiedSetData.txt")
training_set = classified_set

# number of hidden nodes = number of features
num_input_nodes = len(classified_set[0][1])
num_hidden_nodes = num_input_nodes
num_output_nodes = 8

forward_propagation()
