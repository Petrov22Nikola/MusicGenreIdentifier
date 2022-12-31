# Tensorflow Convolutional Neural Network
# (0) Split Train / Test Data
# (1) Build Model
# (2) Compile Model
# (3) Train Model
# (4) Evaluate Model
# (5) Make Predictions

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataDir = "genreData.json"

def loadData(dataPath):
    with open(dataPath, "r") as fp:
        data = json.load(fp)
    
    # Convert list -> NumPy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

# Split data -> Train & Validation & Test
def TrainValTestSplit(testSize, valSize):
    # Load data
    inputs, targets = loadData(dataDir)
    # Split data -> Train & Test
    xTrain, xTest, yTrain, yTest = train_test_split(inputs, targets, test_size = testSize)
    # Split Test Data -> Test & Validation
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = valSize)
    # Reshape -> 3D Data (Adding single channel parameter)
    xTrain = xTrain[..., np.newaxis]
    xVal = xVal[..., np.newaxis]
    xTest = xTest[..., np.newaxis]
    return xTrain, xVal, xTest, yTrain, yVal, yTest

xTrain, xVal, xTest, yTrain, yVal, yTest = TrainValTestSplit(0.25, 0.2)

# Build convolutional neural network
model = tf.keras.Sequential()
# (1) Convolutional Layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
model.add(tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = 'same'))
model.add(tf.keras.layers.BatchNormalization()) # Normalizes activations -> Speeds up training & increases reliability

# (2) Convolutional Layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
model.add(tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = 'same'))
model.add(tf.keras.layers.BatchNormalization()) # Normalizes activations -> Speeds up training & increases reliability

# (3) Convolutional Layer
model.add(tf.keras.layers.Conv2D(32, (2, 2), activation = 'relu', input_shape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides = (2, 2), padding = 'same'))
model.add(tf.keras.layers.BatchNormalization()) # Normalizes activations -> Speeds up training & increases reliability

# Flatten & Feed -> Dense Layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.3))

# Output Layer
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# Compile Model
opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train Model
model.fit(xTrain, yTrain, validation_data = (xVal, yVal), batch_size = 32, epochs = 40)

# Evaluate Model
err, acc = model.evaluate(xTest, yTest)
print("Accuracy: " + str(acc))

# Predict
sampleMusic = xTest[100]
sampleAns = yTest[100]
sampleMusic = sampleMusic[np.newaxis, ...]
prediction = model.predict(sampleMusic)
print(prediction)
print("Genre: " + str(sampleAns))
