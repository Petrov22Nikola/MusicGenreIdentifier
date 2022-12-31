# MusicGenreIdentifier  
Identifies the musical genre of a song, deploys neural networks via TensorFlow and Keras  
  
Performance Summary:  
Convolutional Neural Network: 72.0% accuracy  
Standard Neural Network: 57.9% accuracy  
  
Convolutional Neural Network:  
Epoch 40/40  
188/188 [==============================] - 4s 24ms/step - loss: 0.5964 - accuracy: 0.7888 - val_loss: 0.8331 - val_accuracy: 0.7203  
79/79 [==============================] - 1s 7ms/step - loss: 0.8483 - accuracy: 0.7201  
Accuracy: 0.720064103603363  
  
Neural Network:  
  
Epoch 50/50  
2s 8ms/step - loss: 0.2085 - accuracy: 0.9313 - val_loss: 2.6356 - val_accuracy: 0.5791  
![model](https://user-images.githubusercontent.com/73067824/209717989-f97d99ac-cbdb-47c2-8135-8549fd8479e8.png)  
  
Encountered an overfitting issue where the discrepancy between training set data accuracy and test data accuradcy reached ~30%:  
![overfitting](https://user-images.githubusercontent.com/73067824/209717458-7257c76c-fef0-4c65-9034-6104659e91bb.png)  
   
Solved the overfitting issue by implementing neuron dropout and L2 regularization:  
![overfittingSolution](https://user-images.githubusercontent.com/73067824/209717664-8dca8b0d-4718-4ea1-8e3e-a6c2271eb484.png)  
