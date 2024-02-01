# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split

"""Code to train a CNN model, which classifies a given facial expression dataset."""

df = pd.read_csv("/content/facialex.csv")  # this dataset is the FER2013, a facial expression dataset released by Kaggle in 2013
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
df

random.seed(2023)

"""Prepare data"""

X = df.iloc[:, 1:]
y = df.iloc[:, 0]
print(X.head())
print(y.head())

"""Visualize the first 25 images from the dataset to ensure the import has been done successfully"""

plt.figure(figsize=[8, 8])

for i in range(30):
  plt.subplot(5, 6, i+1)
  picture = X.iloc[i]  # extracting the index and placing it in picture
  picture = np.array(picture)  # converting the picture into an array, and then reshaping to visualize the output
  picture = picture.reshape(48, 48)

  plt.imshow(picture, cmap='gray')  # render the picture directly as it is tensor
  plt.title(class_names[y[i]])  # mapping the index to get the actual letter
  plt.axis('off')

"""Prepare the data to build the Convolutional Neural Network"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)  # Partition the dataset into train and test sets.

# Scale all excel values - dividing by 255 as images are in grayscale, from 0-255 to allow shading, which removes rough edges.
X_train = X_train / 255
X_test = X_test / 255

X_train = np.reshape(X_train.to_numpy(), (-1, 48, 48))  # Reshape X_train to prepare it for CNN
X_test = np.reshape(X_test.to_numpy(), (-1, 48,48))  # Reshape  X_test to prepare it for CNN
print("The shape of the feature train set is:", X_train.shape)
print("The shape of the target train set is:", y_train.shape)
print("The shape of the feature test set is:", X_test.shape)
print("The shape of the target test set is:", y_test.shape)

""" Build a CNN Model"""
model = Sequential()

# The first conv layer’s neurons are not connected to every single pixel in the input layer, instead only
# to pixels in their receptive fields. (first layer of pixels stays square, not flattened).
model.add(Conv2D(50, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', input_shape=(48, 48, 1)))
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

# Final layer is the output layer with softmax. 7 possible classifications.
model.add(Dense(7, activation='Softmax'))

model.summary()

"""Compile the model"""
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Train the model"""

history = model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test))

"""Plot the loss and accuracy curves for both train and validation sets to ensure successful model improvement over time"""

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

"""Visualize the first 25 images of the test partition to ensure accuracy"""

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # turning y_pred from 2d to 1d

plt.figure(figsize= (14, 14))

for i in range (25):
  plt.subplot(5, 5, i+1)
  picture = X.iloc[i]  # extracting the index and placing it in picture
  picture = np.array(picture)  # converting the picture into an array, and then reshaping to visualize the output
  picture = picture.reshape(48, 48)
  plt.imshow(picture, cmap='gray')  # render the picture directly as it is tensor
  plt.title('True: %s, \nPredicted: %s' %
                (class_names[np.array(y_test)[i]], class_names[y_pred[i]]))
  plt.axis('off')

plt.subplots_adjust(hspace=1)

"""Visualize the predicted and actual image labels for 25 random misclassified images"""

# Dealing with tensorflow shape rather than DF. Therefore, approach is different:
all_failed_indices = []
iteration = 0

for i in y_test:  # finding all failed/misclassified indices
  if i != y_pred[iteration]: all_failed_indices.append(iteration)
  iteration = iteration + 1

plt.figure(figsize=(14, 14))

for j in range(25):
    plt.subplot(5, 5, j + 1)
    random = np.random.randint(0, len(all_failed_indices))  # pick a random failed prediction from the TF
    failed_index = all_failed_indices[random]

    failed_sample = X_test[failed_index]
    plt.imshow(failed_sample, cmap='gray')
    plt.title('True: %s, \nPredicted: %s' %
                (class_names[np.array(y_test)[failed_index]], class_names[y_pred[failed_index]]))
    plt.axis('off')
plt.show()
