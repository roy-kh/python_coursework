# -*- coding: utf-8 -*-

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt

"""Write code to train a CNN model which classifies the CIFAR100 dataset."""

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

# print("The shape of the feature train set is:", X_train.shape)
# print("The shape of the target train set is:", y_train.shape)
# print("The shape of the feature test set is:", X_test.shape)
# print("The shape of the target test set is:", y_test.shape)

class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle'
, 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle'
, 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur'
, 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard'
, 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain'
, 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree'
, 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea'
, 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower'
, 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle'
, 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']  # array of labels

"""Visualize the first 30 images from the train dataset"""

plt.figure(figsize=[8, 8])

for i in range(30):
  plt.subplot(5, 6, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(X_train[i])  # No need to reshape X_train. X is already a tensorflow, can index it regularly
  plt.xlabel(class_names[y_train[i, 0]])  # y_train is 2D, must index it regularly

"""Build a CNN model to classify the CIFAR100 dataset"""

# Scale all excel values - dividing by 255 as images are in grayscale, from 0-255 to allow shading, which removes rough edges.
X_train = X_train / 255
X_test = X_test / 255

model = Sequential()

# The first conv layer’s neurons are not connected to every single pixel in the input layer, instead only
# to pixels in their receptive fields. (first layer of pixels stays square, not flattened).
model.add(Conv2D(50, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(75, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
# filter= 50  - the number of output filters in the convolution.
# kernel_size= specifying the height and width of the 2D convolution window.
# Strides= parameter dictating kernel movement across the input data.
# input_shape= check shape in variables

# MaxPool layer is added to shrink the input image, in order to reduce the computation size.
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(75, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))

# One way to prevent overfitting in NN is to ignore randomly selected neurons during training
# In other words, “drop” some neurons. Here, we drop 25% of the neurons.
model.add(Dropout(0.25))
model.add(Conv2D(100, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# After convolution/pooling, image gets smaller and deeper (more feature maps).
# Finally, a regular feedforward NN is added of fully connected layers (and ReLU).  These are flattened layers.
model.add(Flatten())
# Dense layers capture complex patterns in the data and learn the relationships between different parts of the input
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

# Final layer is the output layer with softmax. 100 possible classifications.
model.add(Dense(100, activation='Softmax'))

model.summary()

"""Compile the model"""

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Train the model"""

history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

"""Plot the loss and accuracy curves for both train and validation sets"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy', 'Validation Accuracy'])  # validation accuracy = test accuracy
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")

"""Visualize the first 30 images in the dataset"""

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # turning y_pred from 2d to 1d

plt.figure(figsize= (14, 14))

for i in range (30):
  plt.subplot(5, 6, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(X_test[i])
  plt.title('True: %s, \nPredicted: %s' %
                (class_names[y_test[i, 0]], class_names[y_pred[i]]))

plt.subplots_adjust(hspace=1)

"""Visualize 30 random misclassified images."""

# Dealing with tensorflow shape rather than DF. Therefore, approach is different:
all_failed_indices = []
iteration = 0

for i in y_test:  # finding all failed/misclassified indices
  if i != y_pred[iteration]: all_failed_indices.append(iteration)
  iteration = iteration + 1

plt.figure(figsize=(14, 14))

for j in range(30):
    plt.subplot(5, 6, j + 1)
    random = np.random.randint(0, len(all_failed_indices))  # pick a random failed prediction from the TF
    failed_index = all_failed_indices[random]

    failed_sample = X_test[failed_index]
    plt.imshow(failed_sample, cmap='gray')
    plt.title('True: %s, \nPredicted: %s' %
                (class_names[y_test[failed_index, 0]], class_names[y_pred[failed_index]]))
    plt.axis('off')
plt.show()
