#this is a basic libary to create and use neural networks in python
import numpy as np

from network import Network
from src.pythonNeuralNetwork.Layers.fcLayer import FCLayer
from src.pythonNeuralNetwork.Layers.activationLayer import ActivationLayer
from src.pythonNeuralNetwork.Functions.activationFunctions import tanh, tanh_prime, sigmoid, sigmoid_prime, binary, binary_prime
from src.pythonNeuralNetwork.Functions.lossFunction import mse, mse_prime

#training data
x_train = np.array([[[0, 0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

#network
net = Network()
net.add(FCLayer(2, 5))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(5,1))
net.add(ActivationLayer(tanh, tanh_prime))

#train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

#test
out = net.predict(x_train)
print(out)