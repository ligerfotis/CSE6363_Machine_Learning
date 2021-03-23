import math
import numpy as np

from data import data_x, labels_or, labels_xor


def activate(linear_input):
    return 1 / (1 + np.exp(-linear_input))


class NeuralNetwork:
    def __init__(self, input_shape, layer1_size=4, layer2_size=1):
        self.W1 = np.random.uniform(low=-0.1, high=0.1, size=(input_shape[1], layer1_size))
        self.b1 = np.random.uniform(low=-0.1, high=0.1, size=(1, layer1_size))
        self.W2 = np.random.uniform(low=-0.1, high=0.1, size=(layer1_size, layer2_size))
        self.b2 = np.random.uniform(low=-0.1, high=0.1, size=(1, layer2_size))

        self.z1 = None
        self.z2 = None
        self.layer1 = None
        self.layer2 = None

        self.dloss_dw1 = None
        self.dloss_db1 = None
        self.dloss_dw2 = None
        self.dloss_db2 = None

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.layer1 = activate(self.z1)
        self.z2 = np.dot(self.layer1, self.W2) + self.b2
        self.layer2 = activate(self.z2)
        return self.layer2

    def predict(self, x):
        return self.forward(x)

    def backpropagation(self, sample, label):
        self.forward(sample)

        dloss_da2 = 2 * (label - self.layer2)
        da2_dz2 = self.layer2 * (1 - self.layer2)
        delta_2 = dloss_da2 * da2_dz2
        # dz2_dw2 = self.layer1

        self.dloss_dw2 = np.dot(self.layer1.T, delta_2)
        self.dloss_db2 = delta_2

        dloss_da1 = delta_2.dot(self.W2.T)

        da1_dz1 = self.layer1 * (1 - self.layer1)
        # dz1_dw1 = sample
        delta_1 = dloss_da1 * da1_dz1

        self.dloss_dw1 = np.dot(sample.T, delta_1)
        self.dloss_db1 = delta_1

    def train(self, data_x, labels, epochs, learning_rate=0.01):
        n_samples, n_features = data_x.shape
        for e in range(epochs):
            # randomly pick a sample
            sample_idx = np.random.randint(n_samples)
            sample = data_x[sample_idx]
            label = labels[sample_idx]

            self.backpropagation(np.reshape(sample, (1, 3)), label)

            self.W1 += learning_rate * self.dloss_dw1
            self.b1 += learning_rate * self.dloss_db1
            self.W2 += learning_rate * self.dloss_dw2
            self.b2 += learning_rate * self.dloss_db2


epochs = 1000
lr = 0.001

labels_or = [1 if label == 1 else 0 for label in labels_or]
net_or = NeuralNetwork(np.asarray(data_x).shape)
net_or.train(np.asarray(data_x), np.asarray(labels_or), epochs, lr)
correct_preds_or = 0
for sample in data_x:
    correct_preds_or += 1 if net_or.predict(sample) >= 0.5 else 0

labels_xor = [1 if label == 1 else 0 for label in labels_xor]
net_xor = NeuralNetwork(np.asarray(data_x).shape)
net_xor.train(np.asarray(data_x), np.asarray(labels_xor), epochs, lr)
correct_preds_xor = 0
for sample in data_x:
    correct_preds_xor += 1 if net_xor.predict(sample) >= 0.5 else 0

print("Neural Network Convergence:\nOR: {} correct predictions\nXOR: {} correct predictions".format(correct_preds_or,
                                                                                                    correct_preds_xor))
