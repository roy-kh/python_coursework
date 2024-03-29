
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

df = pd.read_csv("/content/chineseMNIST.csv")  # Read the dataset. It can easily be found online as chineseMNIST is a popular dataset.

X = df.iloc[:, :-2]
y = df.iloc[:, -2]
print(X.head())
print(y.head())

""" Explore the dataset: plot a histogram to visualize the quanity of each Chinese number"""

plt.figure(figsize=[10, 10])

sb.countplot(x=(df['label']), data=df)  # here you can see the frequency of the letters
plt.xticks(rotation=90)
plt.show()

"""Visualize 20 random characters from the train dataset to ensure that it is set up correctly."""

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

font_translation = "/content/SimHei.ttf"  # this translation file that we will use is attached
font = FontProperties(fname=font_translation)

for i in range(20):
    plt.subplot(4, 5, i + 1)
    r = np.random.randint(1, len(X))  # picking a random index from the dataset
    picture = X.iloc[r].values.astype(float)  # extracting the index and placing it in picture
    picture = np.array(picture)  # converting the picture into an array, and then reshaping
    picture = picture.reshape(64, 64)
    plt.title(f"{mapping[y[r]]}\n{y[r]}", fontproperties = font)  # mapping the index to get the actual letter
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
plt.show()

"""Prepare the data to build the Convolutional Neural Network"""
X = X/255  # As you are using a grayscale image, you will need to scale the pixel values by using 255 as divisor

y_copy = df['label'].copy()

# extracting all the unique possible outcomes for y
unique = df['label'].unique()
possible_labels = unique[:15]

# creating the new labels for y, from 0 to 14, and mapping them to the correct output variables
labelConverter = {label: i for i, label in enumerate(possible_labels)}
y_mapped = y_copy.map(labelConverter)

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.3, random_state=2023, stratify=y)  # Partition the dataset into train and test sets.

# print statements below are used to identify the structures of each array which we will work with to build the model
# print("The shape of the feature train set is:", X_train.shape)
# print("The shape of the target train set is:", y_train.shape)
# print("The shape of the feature test set is:", X_test.shape)
# print("The shape of the target test set is:", y_test.shape)

"""Build a model of the NN using Keras layers"""

model = keras.models.Sequential()

# Adding our input 'flatten' layer, with the role to to convert each input image into a 1D array.
model.add(keras.Input(shape=(4096,)))  # The shape specified as 784 as our input has 784 pixels. This instantiates keras tensors

model.add(keras.layers.Dense(100, activation="relu"))  # adding hidden (dense/fully connected) layers
model.add(keras.layers.Dense(100, activation="relu"))  # 2 hidden (dense/fully connected) layers

# Dense 'output' layer with 15 neurons - 1 per classification, using the softmax activation function as the classes are exclusive
model.add(keras.layers.Dense(15, activation="softmax"))


model.summary()

"""Compile the model"""

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Train the model"""

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

"""Plot loss and accuracy curves for both train and test partitions to ensure successful model improvement over time"""

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

"""Print the confusion matrix to ensure accuracy"""

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1), labels=np.unique(y_test))
print(cm)

"""Visualize the first 16 images in the dataset and ensure accuracy"""

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

"""Visualize 1 random misclassified image. What number did the model think it was, and what number was it actually?"""

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
