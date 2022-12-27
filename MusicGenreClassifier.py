# Music Genre Classifier Neural Network

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataDir = "Path to dataset"

def loadData(dataPath):
    with open(dataPath, "r") as fp:
        data = json.load(fp)
    
    # Convert list -> NumPy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

# Load data
inputs, targets = loadData(dataDir)
# Split data -> Train & Test
xTrain, xTest, yTrain, yTest = train_test_split(inputs, targets, test_size = 0.3)
# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])), # Input layer
    tf.keras.layers.Dense(512, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.001)), # First hidden layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation = "relu"), # Second hidden layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation = "relu"), # Third hidden layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation = "softmax") # Output layer
])
# Compile neural network
opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.summary()
# Train network
model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs = 50, batch_size = 32)