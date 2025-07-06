import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def load_data():
    train_data=pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Separate features in X and labels in Y
    X_train = train_data.drop('label', axis=1).values
    Y_train = train_data['label'].values
    X_test = test_data.values #Turning the test data into a numpy array

    # Normalizing the data to the range [0, 1] since pixel values are in the range [0, 255]
    # This is important for neural networks to converge faster
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode the labels (doesn't treat them as values but as categories)
    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))

    return X_train, Y_train, X_test, encoder

def ReLU(z): # Activation function(helps introduce non-linearity)
    return np.maximum(0, z)

def ReLU_derivative(z): # Derivative of the ReLU function
    return np.where(z > 0, 1.0, 0.0)

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # subtract max for numerical stability
    return exp_z / np.sum(exp_z)

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # avoid log(0)
    return -np.sum(y_true * np.log(y_pred))

class Neuron:
    def __init__(self, input_size, activation_func, activation_deriv):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * np.sqrt(2. / input_size) # He initialization helps with faster convergence
        self.bias = np.random.rand()
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv

    def forward(self, x):
        self.x=x
        self.z = np.dot(x, self.weights) + self.bias
        a=self.activation_func(self.z)
        return a
    
    def backward(self, error, learning_rate):
        activation_derivative = self.activation_deriv(self.z)
        delta = error * activation_derivative #What is the error and how much does it affect the output
        delta = np.clip(delta, -1, 1) # Clipping to prevent exploding gradients
        for i in range(len(self.weights)): #Update the weights
            self.weights[i] -= learning_rate * delta * self.x[i] #Weights depend on the input
        self.bias -= learning_rate * delta
        return delta * self.weights # Return the error for the previous layer

class Layer:
    def __init__(self, num_neurons, input_size, activation_func, activation_deriv):
        # Initialize a layer with a specified number of neurons
        self.neurons = []
        for i in range(num_neurons):
            neuron=Neuron(input_size=input_size, activation_func=activation_func, activation_deriv=activation_deriv)
            self.neurons.append(neuron)

    def forward(self, input_data):
        self.x = input_data  # Store input for use in backward pass
        outputs=[]
        for neuron in self.neurons:
            output= neuron.forward(input_data)
            outputs.append(output)
        return np.array(outputs)
    
    def backward(self, error, learning_rate):
        input_error = np.zeros(len(self.x)) # Initialize input error
        for i, neuron in enumerate(self.neurons):
            delta= neuron.backward(error[i],learning_rate)
            input_error += delta
        return input_error

class Network:
    def __init__(self, layer_size):
        self.layers = []
        for i in range(1, len(layer_size)): # Number of hidden layers
            input_size = layer_size[i-1]
            output_size = layer_size[i]
            if i == len(layer_size) - 1:
                activation_func = lambda z : z  # Output layer uses identity activation
                activation_deriv = lambda z : 1  # Derivative is 1
            else:
                activation_func = ReLU
                activation_deriv = ReLU_derivative
            layer = Layer(num_neurons=output_size, input_size=input_size, activation_func=activation_func, activation_deriv=activation_deriv)
            self.layers.append(layer)

    def forward(self, input_data):
        output = input_data # Start with the input data
        for layer in self.layers:
            output = layer.forward(output)
        output = np.clip(output, -50, 50)  # keep logits in safe range
        return softmax(output)


    def backward(self , y_pred , y_true, learning_rate):
        grad_error = y_pred - y_true #cross-entropy loss gradient
        for layer in reversed(self.layers):
            grad_error = layer.backward(grad_error, learning_rate)

def train(network, X_train, Y_train, epochs, learning_rate):
    losses = []

    X_train_split, X_val, Y_train_split, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    for epoch in range(epochs):
        loss=0
        for x, y in zip(X_train_split,Y_train_split):
            output = network.forward(x) # Forward pass
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                print(f"Skipping corrupted sample at epoch {epoch + 1}")
                continue  # Skip this one and move to the next
            # Helps to avoid NaN or Inf values in the output which causes explosions
            loss += cross_entropy_loss(output, y) # Cross-entropy loss calculation
            network.backward(output, y, learning_rate) # Backward pass

        avg_loss = loss / len(X_train_split) # Average loss
        losses.append(avg_loss)
        print("Epoch: ", epoch + 1, "Loss: ",avg_loss)

    correct=0
    for x, y in zip(X_val, Y_val):
        output = network.forward(x)
        predicted_label = np.argmax(output)
        true_label = np.argmax(y)
        if predicted_label == true_label:
            correct += 1
        
    accuracy_percent = (correct / len(X_val)) * 100
    print("Accuracy: ", accuracy_percent, "%")

    plt.plot(losses)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

train_data, test_data = load_data()
X_train, Y_train, X_test, encoder = preprocess_data(train_data, test_data)
network = Network(layer_size=[784, 128, 64, 10]) # Input layer, hidden layer 1, hidden layer 2, output layer

epochs = 30
learning_rate = 0.0003
train(network, X_train, Y_train, epochs, learning_rate)

def generate_submission(network, X_test, filename="submission.csv"):
    predictions = []

    for i, x in enumerate(X_test, start=1):
        output = network.forward(x)
        predicted_label = np.argmax(output)
        predictions.append([i, predicted_label])

    # Create a DataFrame with the required format
    submission_df = pd.DataFrame(predictions, columns=["ImageId", "Label"])
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved as: {filename}")

generate_submission(network, X_test, filename="submission.csv")