# -*- coding: utf-8 -*-
"""
Multi-layer Perceptron Classifier for Spirals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

"""The goal is to train a neural network to do binary classification of spirally distributed 2-dimensional data.

Firstly, I generate x and y coordinates of spirally distributed blobs in 2 seperate colors.
I added some noise to x and y to test the classifier.
"""

n_samples = 500
noise = 0.75

x1 = []  # first spiral x coordinate
y1 = []  # first spiral y coordinate
x2 = []  # second spiral x coordinate
y2 = []  # second spiral y coordinate

for x in range(n_samples):  # First spiral
    theta = 2 * np.pi * x / n_samples
    r = 2 * theta + np.pi
    x1.append(r * np.cos(theta) + np.random.normal(0, noise))
    y1.append(r * np.sin(theta) + np.random.normal(0, noise))

for x in range(n_samples):  # Second spiral
    theta = 2 * np.pi * x / n_samples
    r = -2 * theta - np.pi
    x2.append(r * np.cos(theta) + np.random.normal(0, noise))
    y2.append(r * np.sin(theta) + np.random.normal(0, noise))

X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))  # converting into 2D array columns, then
# stacking the columns into a single column
y = np.hstack((np.zeros(500), np.ones(500)))

print(X)  # coordinates of the 1000 dots
print(y)  # color of the coordinate

# visualize the dots
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu') # X[:, 0] is all rows, only first column - x axis, X[:, 1] is all rows only second column - y axis
# the colors come from the y specified
plt.title('Spiral Data')
plt.show()

"""Create partitions with 70% train dataset. Stratify the split."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

"""Train the network using MLP Classifier from scikit learn."""
model = MLPClassifier(hidden_layer_sizes=(7, 7, 7, 7), activation='relu', max_iter=1000, random_state=2023,
                      learning_rate_init=0.01, solver='adam', alpha=0.001, verbose=True)
# hidden_layer_sizes= 4 layers, 7 neurons each
# activation= your activation function
# max_iter = number of epocs
# alpha = how close does the slope get
# Solver = algorithm for weight optimization
# learning_rate_init= how big of a step to take
# verbose= shows us the response of total loss (training/test loss) throughout testing
model.fit(X_train, y_train)

"""Plot the decision boundary (along with the original spirals)."""
X1 = np.arange(-20, 20, 0.1)  # x coordinate
X2 = np.arange(-20, 20, 0.1)  # y coordinate
X1, X2 = np.meshgrid(X1, X2)  # Generate combinations of X1, X2 that is a mesh to cover the entire field

"""Reshape the meshgrid to a dataframe."""
X_decision = pd.DataFrame({'X0': np.reshape(X1, 160000), 'X1': np.reshape(X2, 160000)})  # 160000 dots created by the mesh(40/0.1 * 40/0.1)

"""classify each point using the trained model (model.predict)"""
y_mesh = model.predict(X_decision)   # Now predict the color of each dot (row) in the dataframe

"""Plot both the original data points (spirals) and the mesh data points to generate a decision boundary"""
plt.scatter(X_decision['X0'], X_decision['X1'], c=y_mesh, cmap='cool')  # this creates the background circular split
# Now visualize each of the predicted colors in y_mesh
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu')
plt.show()
