'''
This implements and tests a naive neural network model 
to determine if two graphs are isomorphic
Input to the neural network are the features used in the naive_model
'''

import networkx as nx
import random
import numpy as np

import sys
sys.path.append('../../')
sys.path.append('../')

import features
import graph_pair_class
import permute_graph
from graphScripts import ping

from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam


# List of features used in this model
feature_list = [features.CompareNumOfNodes, features.CompareNumOfEdges, features.CompareDirected,
                features.CompareDegreeDistribution, features.CompareLSpectrum, features.CompareASpectrum]

# All graph pairs are contained in this list
graphPairs = []

# Base graph for all isomorphic graphs
g_base_nodes = 12
g_base = nx.watts_strogatz_graph(g_base_nodes, 5, 0.1)

# Create a list of isomorphic graph pairs
print ("Creating list of isomorphic graph pairs...")
for i in range(100):
    g = permute_graph.permute_graph(g_base, random.randint(1,g_base_nodes))
    graphPair = graph_pair_class.GraphPair(g_base, g)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")

# Create a list of random graph pairs
print ("Creating list of random graph pairs...")
for i in range(100):
    g = nx.watts_strogatz_graph(g_base_nodes, 5, 0.1)
    graphPair = graph_pair_class.GraphPair(g_base, g)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")

# Create a list of PING graphs
print ("Creating PING graphs...")
for i in range(100):
    g1, g2 = ping.create(2,10)
    graphPair = graph_pair_class.GraphPair(g1,g2)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")


'''Neural Network model'''
X = np.array([g.features for g in graphPairs])
y = np.array([g.is_isomorphic for g in graphPairs])
shuffle = list(range(len(y)))
random.shuffle(shuffle)
X = X[shuffle]
y = y[shuffle]

# define the Neural Network model
model = Sequential()
model.add(Dense(output_dim=10, input_dim=6, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=1, activation='sigmoid'))

# sgd  = SGD()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# model.compile(optimizer=sgd, loss='mean_absolute_error')
# model.compile(optimizer=adam, loss='mean_absolute_error')
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

# training on train set
# model.fit(X, y, nb_epoch=20, verbose=2, validation_split=0.2)
model.fit(X, y, nb_epoch=500, batch_size=10,validation_split=0.2)

train_scores = model.evaluate(X, y)

model.save("naive_nn_model.model")



'''test the model on test set'''
# create test set
# All graph pairs are contained in this list
graphPairs = []

# Base graph for all isomorphic graphs
g_base_nodes = 20
g_base = nx.watts_strogatz_graph(g_base_nodes, 5, 0.1)

# Create a list of isomorphic graph pairs
print ("Creating list of isomorphic graph pairs...")
for i in range(100):
    g = permute_graph.permute_graph(g_base, random.randint(1,g_base_nodes))
    graphPair = graph_pair_class.GraphPair(g_base, g)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")

# Create a list of random graph pairs
print ("Creating list of random graph pairs...")
for i in range(100):
    g = nx.watts_strogatz_graph(g_base_nodes, 5, 0.1)
    graphPair = graph_pair_class.GraphPair(g_base, g)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")

# Create a list of PING graphs
print ("Creating PING graphs...")
for i in range(100):
    g1, g2 = ping.create(2,10)
    graphPair = graph_pair_class.GraphPair(g1,g2)
    for f in feature_list:
        graphPair.add_feature(f)
    graphPairs += [graphPair]
print ("Done!")

X_test = np.array([g.features for g in graphPairs])
y_test = np.array([g.is_isomorphic for g in graphPairs])
shuffle = list(range(len(y)))
random.shuffle(shuffle)
X_test = X_test[shuffle]
y_test = y_test[shuffle]

# test on model
test_scores = model.evaluate(X_test, y_test)


'''show the two results'''
print("train %s: %.2f%%" % (model.metrics_names[1], train_scores[1]*100))
print("test %s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100))