import numpy as np
import pandas as pd

from data import heights, weights, age, gender, samples, k_values


def cartesian_distance(sample, inputs):
    """
    :param sample
    :param inputs: a dataset
    :return: the cartesian distances between a sample and the dataset
    """

    diffs = sample - inputs
    sum_pow = np.sum(np.power(diffs, 2), axis=1)

    return np.power(sum_pow, 0.5)


def classify(k, sorted_labels):
    """
    Performs classification using the k nearest neighbors on a list of sorted labels
    :param k: number of neighbors
    :param sorted_labels: sorted array of labels based on descending cartesian distance
    :return: predicted class
    """
    k_neighbors = sorted_labels[:k]
    men_occurencies = np.count_nonzero(k_neighbors == 'M')
    women_occurencies = np.count_nonzero(k_neighbors == 'W')

    return 'M' if men_occurencies > women_occurencies else 'W'


def KNN_classification(sample, k, df_dataset, drop_age):
    """
    Classifies a sample using KNN on a given dataset
    :param sample: a sample to classify
    :param k: number of neighbors
    :return: predicted class
    """
    if drop_age:
        inputs = df_dataset.drop(['age', 'gender'], axis=1).values
    else:
        inputs = df_dataset.drop(['gender'], axis=1).values

    labels = df_dataset["gender"].values

    cart_distance = cartesian_distance(sample, inputs)
    labeled_cart = np.vstack((cart_distance, labels))
    sorted_cart = labeled_cart.T[labeled_cart.T[:, 0].argsort()]
    sorted_labels = sorted_cart.T[1]

    return classify(k, sorted_labels)


if __name__ == '__main__':
    df_dataset = pd.DataFrame({'heights': heights, 'weights': weights, 'age': age, 'gender': gender})

    for sample in samples:
        for k in k_values:
            prediction_1 = KNN_classification(sample, k, df_dataset, drop_age=False)
            print("Prediction is {} for k:{} number of neighbors".format(prediction_1, k))

            prediction_2 = KNN_classification(sample[:2], k, df_dataset, drop_age=True) # assumption: gender is is the 3rd element of the sample
            print("Prediction is {} for k:{} number of neighbors without using age feature".format(prediction_2, k))

        print()
