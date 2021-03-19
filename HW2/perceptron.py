import numpy as np

from data import labels_or, data_x, labels_xor


def activate(linear_input):
    return np.sign(linear_input)


class Perceptron:
    def __init__(self):
        self.bias = 1
        self.weights = None

    def predict(self, sample):
        linear = self.weights.dot(sample) + 1
        return activate(linear)

    def fit(self, data_x, labels, epochs, learning_rate):
        n_samples, n_features = data_x.shape
        self.weights = np.zeros(n_features) # initialize weights

        correct_predictions = 0
        for e in range(epochs):
            # randomly pick a sample
            sample_idx = np.random.randint(n_samples)
            sample = data_x[sample_idx]
            label = labels[sample_idx]

            predicted = self.predict(sample)
            delta_w = learning_rate * (label - predicted)
            correct_predictions += 1 if label == predicted else 0
            self.weights += delta_w * sample
            self.bias += delta_w

            print("Sample picked: {} (label:{}). \nDelta W: {}. \tW: {}".format(sample, label, delta_w, self.weights))
        return correct_predictions


perceptron_or = Perceptron()
correct_preds_or = perceptron_or.fit(np.asarray(data_x), np.asarray(labels_or), 10, 0.01)

perceptron_xor = Perceptron()
correct_preds_xor = perceptron_xor.fit(np.asarray(data_x), np.asarray(labels_xor), 10, 0.01)

print("\nPerceptron Convergence:\nOR: {} correct predictions\nXOR: {} correct predictions".format(correct_preds_or, correct_preds_xor))