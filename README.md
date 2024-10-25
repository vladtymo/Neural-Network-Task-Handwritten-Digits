# Sprint 16: Introduction to Neural Networks

## Task 1. Building a Neural Network to Classify Handwritten Digits

Create, train, and evaluate a neural network using Python to classify images of handwritten digits from the MNIST dataset. You'll use the Keras library to build a feedforward neural network with one hidden layer.

**Tasks:**

1. **Setup Environment:**
   - Make sure you have the required libraries installed. You'll need:
     - tensorflow
     - keras
     - numpy
     - matplotlib

2. **Load the MNIST Dataset:**
   - The MNIST dataset is a classic dataset of 28x28 grayscale images of handwritten digits (0-9). Keras provides this dataset built-in.

3. **Preprocess the Data:**
   - You'll need to flatten the 28x28 images into a single 784-dimensional vector and one-hot encode the labels.

4. **Build the Neural Network:**
   - Create a simple feedforward neural network using Keras. The network will have:

     - An input layer (784 nodes, corresponding to the flattened image)
     - One hidden layer with 128 neurons and ReLU activation
     - An output layer with 10 neurons (one for each digit, 0-9) using the softmax activation function

5. **Train the Neural Network:**
   - Fit the model to the training data for 10 epochs with a batch size of 32.

6. **Evaluate the Model:**
   - After training, evaluate the model on the test dataset to see how well it performs.

7. **Visualize the Training Process:**
   - Plot the training and validation accuracy over epochs to see how the model improves.

8. **Make Predictions:**
   - Use the trained model to make predictions on the test set and visualize a few examples.

**Expected Result:**

<img width="300" src="https://github.com/vladtymo/Neural-Network-Task-Handwritten-Digits/blob/master/images/digit_4.png" alt="Digit 4 Predict">
<img width="300" src="https://github.com/vladtymo/Neural-Network-Task-Handwritten-Digits/blob/master/images/digit_2.png" alt="Digit 2 Predict">
