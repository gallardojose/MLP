# Author: Hitesh Bhavsar, Jose Gallardo, Amish Mathur, Mike Mai
# This file contains all the code required for the project: from reading an input ClassifiedDataSet file to building a MLP architecture, determining the hidden
# layers nodes, initial link weights. The file also has all the code that outputs the accuracy, number of true postives, false positives, true negatives, and 
# false negatives, and other rates such as error rate, precision rate, recall rate and others of the final MLP on each of the Holdout vectors and Validation Set.

and giving the classification of the decision tree based on thier computed accurracies.
# All functions in this file also have comments explaining in detail the purpose of thier use and what they intended to do when called given the correct parameters.

import re
import math
import random
import numpy as np

Classified_Set = []
Training_Set = []
Holdout_Set = []
Validation_Set = []
# ====================================================
# Code to find out the metrics and confusion Matrix
TruePositive=[0,0,0,0,0,0,0,0]
TrueNegative=[0,0,0,0,0,0,0,0]
FalsePositive=[0,0,0,0,0,0,0,0]
FalseNegative=[0,0,0,0,0,0,0,0]
# ====================================================


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


def sets_generator():
    total_data = len(Classified_Set)
    data_indexes = list(range(total_data))

    total_training_set_data = int(total_data * (.80));
    remaining_data = total_data - total_training_set_data

    Training_indexes = random.sample(data_indexes, total_training_set_data)

    for element in Training_indexes:
        Training_Set.append(Classified_Set[element])
        data_indexes.remove(element)

    Holdout_indexes = random.sample(data_indexes, int(remaining_data))

    for element in Holdout_indexes:
        Holdout_Set.append(Classified_Set[element])
        data_indexes.remove(element);

    
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
    # calc logistic swixi in hidden nodes
    for i in range(num_hidden_nodes):
        hidden_layer.append(logistic(swixi(W1[i], training_vec)))

    # hidden layer to ouput layer
    output_layer = []
    # calc logisitc for output layer
    for i in range(num_output_nodes):
        output_layer.append(logistic(swixi(W2[i], hidden_layer)))

    return hidden_layer, output_layer


def back_propagation(learning_rate, min_accuracy):
    epochs = 0
    while accuracy(Training_Set) < min_accuracy:
        epochs += 1
        for value in Training_Set:
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
    return epochs

def accuracy(set):
    correct = 0
    for data in set:
        hl, ol = forward_propagation(normal(data[1], max_feature_value))
        if (np.argmax(ol) == data[2]):
            correct += 1
    return correct / len(set)

# ====================================================
# code added by hitesh for calculating the confusion matrix and the other metics
leng_of_matrix=8
def accuracy_metrics(set):
    correct = 0
    #print("="*80)
    for data in set:
        hl, ol = forward_propagation(normal(data[1], max_feature_value))
        #print("argmax data")
        index_of_max=np.argmax(ol)
        #print(index_of_max)
        #print("Class ")
        #print(data[2])
        if(np.argmax(ol) == data[2]):
             correct += 1
             index=data[2]
             TruePositive[index]+=1
             for i in range(leng_of_matrix):
                 if(i!=index):
                     TrueNegative[i]+=1
        else:
             predicted_output=index_of_max
             actual_output=data[2]
             FalsePositive[predicted_output]+=1
             FalsePositive[actual_output]+=1
             update_except=[predicted_output,actual_output]
             for i in range(leng_of_matrix):
                 if i not in update_except:
                     TrueNegative[i]+=1
    return correct / len(set)

def clearmatix():
    TruePositive=[0 for i in range(8)]
    TrueNegative=[0 for i in range(8)]
    FalsePositive=[0 for i in range(8)]
    FalseNegative=[0 for i in range(8)]

def printmetrix():
    for i in range(8):
        print("class "+str(i+1))
        tp=TruePositive[i]
        fp=FalsePositive[i]
        tn=TrueNegative[i]
        fn=FalseNegative[i]
    
        print("True Positive: "+str(TruePositive[i]))
        print("False Positive: "+str(FalsePositive[i]))
        print("True Negative: "+str(TrueNegative[i]))      
        print("False Negative: "+str(FalseNegative[i]))

        print("Accuracy: "+str((tp+tn)/(tp+tn+fp+fn)))
        print("Error Rate: "+str((fp+fn)/(tp+tn+fp+fn)))
        print("Precision: "+str((fp+fn)/(tp+tn+fp+fn)))
        print("Recall: "+str((fp+fn)/(tp+tn+fp+fn)))

#========================================================================
        
Classified_Set = read_file("ClassifiedSetData.txt")
sets_generator()

# number of hidden nodes = number of features
num_input_nodes = len(Training_Set[0][1])
num_hidden_nodes = num_input_nodes
num_output_nodes = 8
max_feature_value = 96.0

W1 = init_weights(num_hidden_nodes, num_input_nodes) # weight matrix from input to hidden layer
W2 = init_weights(num_output_nodes, num_hidden_nodes) # weight matrix from hidden to output layer
print("The MLP architecture is as follows. The input nodes connect to 10 hidden layer nodes which then connect to the 8 "
      "output nodes through weights.")
print("Initial weights\nFirst Layer: " + str(W1) + "\nSecond Layer: " + str(W2))
clearmatix()
print("Calculating epoch...")
print("Epochs required: " + str(back_propagation(0.2, .98)))

print("Second Layer part has ended")
print("Final weights\nFirst Layer: " + str(W1) + "\nSecond Layer: " + str(W2))


print("Holdout Accuracy: " + str(accuracy_metrics(Holdout_Set)))
#print("Training Accuracy: " + str(accuracy(Training_Set)))

print("Print matrix and metrics")
print(printmetrix())


    
