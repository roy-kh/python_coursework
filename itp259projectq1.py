# -*- coding: utf-8 -*-
"""ITP259ProjectQ1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LXbrP4lEQqUTGPJBQoxq8yhAXXImCqYb
"""

# Roy Hayyat
# ITP 259 2023
# Final Project
# Problem 1

"""Train a neural network to classify Chinese numbers"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb  # visualization package
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont

"""1. Read in the data (1)"""

df = pd.read_csv("/content/chineseMNIST.csv")

X = df.iloc[:, :-2]
y = df.iloc[:, -2]
print(X.head())
print(y.head())

"""2. Plot the count (histogram) of each Chinese number (1)"""

plt.figure(figsize=[10, 10])

sb.countplot(x=(df['label']), data=df)  # here you can see the frequency of the letters
plt.xticks(rotation=90)
plt.show()

"""3. Visualize 25 random characters from the train dataset. Be sure that the plot shows both the English number and the Chinese number as shown. Hint given."""

mapping = {0: "零",
           1: "一",
           2: "二",
           3: "三",
           4: "四",
           5: "五",
           6: "六",
           7: "七",
           8: "八",
           9: "九",
           10: "十",
           100: "百",
           1000: "千",
           10000: "万",
           100000000: "亿"
           }

plt.figure(figsize=(14,14))

font_translation = "/content/SimHei.ttf"
font = FontProperties(fname=font_translation)

for i in range(25):
    plt.subplot(5, 5, i + 1)
    r = np.random.randint(1, len(X))  # picking a random index from the dataset
    picture = X.iloc[r].values.astype(float)  # extracting the index and placing it in picture
    picture = np.array(picture)  # converting the picture into an array, and then reshaping
    picture = picture.reshape(64, 64)
    plt.title(f"{mapping[y[r]]}\n{y[r]}", fontproperties = font)  # mapping the index to get the actual letter
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
plt.show()

"""4. Scale the pixel values (1)"""

X = X/255  # As you are using a grayscale image, you will use 255 as divisor

"""5. Partition the dataset into train and test sets. Print the shapes of the train and test data sets (1)"""

y_copy = df['label'].copy()

# extracting all the unique possible outcomes for y
unique = df['label'].unique()
possible_labels = unique[:15]

# creating the new labels for y, from 0 to 14, and mapping them to the correct output variables
labelConverter = {label: i for i, label in enumerate(possible_labels)}
y_mapped = y_copy.map(labelConverter)

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.3, random_state=2023, stratify=y)

print("The shape of the feature train set is:", X_train.shape)
print("The shape of the target train set is:", y_train.shape)
print("The shape of the feature test set is:", X_test.shape)
print("The shape of the target test set is:", y_test.shape)

"""6. Build a model of the NN using Keras layers. The type, number and hyperparameters of the layers is up to you (3)"""

model = keras.models.Sequential()

# Adding our input 'flatten' layer, with the role to to convert each input image into a 1D array.
model.add(keras.Input(shape=(4096,)))  # The shape specified as 784 as our input has 784 pixels. This instantiates keras tensors

model.add(keras.layers.Dense(100, activation="relu"))  # adding hidden (dense/fully connected) layers
model.add(keras.layers.Dense(100, activation="relu"))  # 2 hidden (dense/fully connected) layers

# Dense 'output' layer with 15 neurons - 1 per classification, using the softmax activation function as the classes are exclusive
model.add(keras.layers.Dense(15, activation="softmax"))

"""7. Display the model summary (1)"""

model.summary()

"""8. Use the loss function sparse_categorical_crossentropy when compiling the model (1)"""

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""9. Train the model with at least 50 epochs"""

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

"""10. Plot the loss and accuracy curves for both train and test partitions"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Loss Curves")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy', 'Validation Accuracy'])  # validation accuracy = test accuracy
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Accuracy Curves")

"""11. Print the confusion matrix"""

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1), labels=np.unique(y_test))
print(cm)

"""12. Visualize the predicted and actual image labels for the first 16 images in the dataset (4)"""

y_pred = np.argmax(y_pred, axis=1)  # turning y_pred from 2d to 1d

plt.figure(figsize= (14, 14))

for i in range (16):
  plt.subplot(4, 4, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(X_test.iloc[i].values.reshape(64, 64), cmap='gray')
  plt.title('True: %s, \nPredicted: %s' %
                (y_test.iloc[i], y_pred[i]))

plt.subplots_adjust(hspace=1)

"""13. Visualize 1 random misclassified image. Display both the predicted and actual image labels. Also display the Chinese character as the X label (4)"""

# Filter the test dataframe to those cases where the prediction failed
failed_df = X_test[y_pred != y_test]

# get a random sample from the failed df and get its index
failed_index = failed_df.sample(n=1).index
print(failed_index)

# now, 'unflatten' the row at the failed index by converting the series into an array, and then reshaping it
failed_sample = np.array(X_test.iloc[failed_index]).reshape(64, 64)

# plot the incorrectly predicted digit, showing its predicted and actual values:
# y_pred is already a 1d array. y_test is a series however so we need .values to access the index.
plt.imshow(failed_sample, cmap='gray')
plt.title("The actual digit is " + str(y_test[failed_index].values) +
          ". The predicted label is " + str(y_pred[failed_index]))