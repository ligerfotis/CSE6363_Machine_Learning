import numpy as np
from hw1_data import heights, weights, age, gender, samples
import math

learning_rate = 0.00001
epoch_num = 100000

"""""""""""
Question a
"""""""""""


# the logistic function
def sigmoid(theta, input):
    """
    Calculates the logistic function of theta * input
    :param theta: models parameters
    :param input: inout 3D vector
    :return: probability of the input
    """
    h = np.dot(theta, input)
    return 1 / (1 + math.exp(-h))


def logistic_regression_fit(train_input):
    """
    Learns the parameters of the logistic regression model
    :param train_input: training data
    :return: learned parameters
    """
    misclassification_error = 0
    print("Training")
    # initialize theta
    theta = np.random.uniform(low=-0.1, high=0.1, size=(train_input.shape[1]))

    # stochastic gradient descent
    for epoch_i in range(epoch_num):

        index = np.random.randint(0, train_input.shape[0])
        # for sample_i in range(train_input.shape[0]):
        # forward pass
        y_hat = sigmoid(theta, train_input[index])
        # the label
        y = 1 if train_labels[index] == 'M' else 0

        # error = np.power(y_hat, y) * np.power((1 - y_hat), 1 - y)

        if y_hat <= 0.5 and y == 1 or y_hat > 0.5 and y == 0:
            misclassification_error += 1

        # stochastic gradient update
        theta += learning_rate * (y - y_hat) * train_input[index]
        # print(theta)

        if (epoch_i + 1) % 100 == 0:
            print("Avg Misclassification Error: {} on epoch {}".format(misclassification_error / 100, epoch_i + 1))
            if misclassification_error / 1000 <= 0.01:
                break
            misclassification_error = 0
    print("End of Training")
    return theta


train_input = np.asarray([heights, weights, age]).T
train_labels = np.asarray(gender)
theta = logistic_regression_fit(train_input)

"""""""""""
Question b
"""""""""""

print("Testing")
for sample in samples:
    y_hat = sigmoid(theta, sample)
    pred_class = "M" if y_hat >= 0.5 else "W"
    print("Predicted class: {}".format(pred_class))
