from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

#Loading training and test data from the mnist dataset
train_data = pd.read_csv("/content/fmnist/-fashion-mnist_train.csv")
test_data = pd.read_csv("/content/fmnist/-fashion-mnist_test.csv")
df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)
train_labels = df_train['label'].to_numpy()
train_images = df_train.iloc[:,1:785]
train_images = np.array(train_images)
test_labels = df_test['label'].to_numpy()
test_images = df_test.iloc[:,1:785]
test_images = np.array(test_images)

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# Feed forward neural network
model = Sequential()
model.add(Flatten())
model.add(Dense(256,kernel_initializer='random_uniform',activation="relu")) #1st input layer
model.add(Dense(128,kernel_initializer='random_uniform',activation="relu")) #hidden layer
model.add(Dense(10,activation="softmax")) #output layer

# train the model using SGD
sgd = SGD(lr=0.001) #learning rate
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) #multi-class cross entropy
clf =model.fit(train_images, train_labels,epochs=100, batch_size=40)

print("Hyper-parameters")
print("Input layer activation: relu")
print("Hidden layer activation: relu")
print("Output layer activation: softmax")
print("Weights intitialization: random uniform")
print("Learning rate: 0.001")
print("Epochs: 100")
print("Batch size: 40")
print("Optimizer: Stochastic Gradient Descent")

# predicting test dataset
predictions = model.predict(test_images, batch_size=40)

#accuracy
print(classification_report(test_labels.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

plt.plot(np.arange(0, 100), clf.history["loss"], label="train_loss")
plt.xlabel('epochs')
plt.ylabel('training loss')
