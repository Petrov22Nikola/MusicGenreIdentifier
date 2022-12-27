# MusicGenreIdentifier
Identifies the musical genre of a song, deploys neural networks via TensorFlow and Keras

Neural Network:


Epoch 50/50
2s 8ms/step - loss: 0.2085 - accuracy: 0.9313 - val_loss: 2.6356 - val_accuracy: 0.5791
![model](https://user-images.githubusercontent.com/73067824/209717989-f97d99ac-cbdb-47c2-8135-8549fd8479e8.png)


Encountered an overfitting issue where the discrepancy between training set data accuracy and test data accuradcy reached approximately 30%:
![overfitting](https://user-images.githubusercontent.com/73067824/209717458-7257c76c-fef0-4c65-9034-6104659e91bb.png)

Solved the overfitting issue by implementing neuron dropout and L2 regularization:
![overfittingSolution](https://user-images.githubusercontent.com/73067824/209717664-8dca8b0d-4718-4ea1-8e3e-a6c2271eb484.png)
