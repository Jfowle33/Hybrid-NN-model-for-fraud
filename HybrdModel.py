# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:51:18 2022

@author: joefo

The goal of this deep learning model is to reduce the dimensionality of a dataset on credit cards
and then use an ANN to rank the highest probability of fraud by customer id.
"""

"""Part 1 - SOM"""

"""### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv("D:\P16-Self-Organizing-Maps\Self_Organizing_Maps\Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds"""


mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,8)], mappings[(5,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)


"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
  
  
"""Part 2 - Going from Unsupervised to Supervised Deep Learning"""

"""Create Matrix of Features"""

customers = dataset.iloc[:, 1:].values

"""Create Dependent Variable"""

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1
    
"""Part 3 - ANN"""
"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

import tensorflow as tf
tf.__version__


# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #rectified linear activation function
#f(x) = max(0,x), more computationally efficient
#units-number of neurons (input layer)

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #rectified linear activation function
#ReLU function does not activate all the neurons at the same time. 

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #Logistic
#f(x) = 1 / 1+e^-x

# Part 3 - Training the ANN

# Compiling the ANN
#Adam is a stochastic gradient descent optimization method
'''
Loss function: Entropy is a measure of the uncertainty associated with
a given distribution q(y).
    
H_p(q) = -1/n /sum^n_{i=1} y_i * log(p(y_i)) + (1-y_i) * log(1-p(y_i))

If we compute entropy like above, we are actually computing the cross-entropy 
between both distributions.

cross-entropy will have a BIGGER value than the entropy computed on the true 
distribution.
 
 H_p(q)-H(q) >= 0
 
Daniel Godoy-Understanding binary cross-entropy / log loss: a visual explanation
'''
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(customers, is_fraud, batch_size = 1, epochs = 10)

y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]#sorting by column 1


print(y_pred)
#First column is the customer id, 2nd column is the predicted probabilities of fraud
