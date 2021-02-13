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

    # get the cartesian distance from each data point
    cart_distance = cartesian_distance(sample, inputs)

    # create a 2D array with the 1st column being the above distances and the second corresponding label
    labeled_cart = np.vstack((cart_distance, labels))

    # sort in an ascending manner the above 2D array based on the distances
    sorted_cart = labeled_cart.T[labeled_cart.T[:, 0].argsort()]
    sorted_labels = sorted_cart.T[1]

    return classify(k, sorted_labels)


if __name__ == '__main__':
    df_dataset = pd.DataFrame({'heights': heights, 'weights': weights, 'age': age, 'gender': gender})

    for sample in samples:
        print("sample:{}".format(sample))
        for k in k_values:
            print("\tK:{}".format(k))
            prediction_1 = KNN_classification(sample, k, df_dataset, drop_age=False)
            print("\tPrediction is {} for k:{} number of neighbors".format(prediction_1, k))
            # prediction_2 = KNN_classification(sample[:2], k, df_dataset,
            #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
            # print("\tPrediction is {} for k:{} number of neighbors without using age feature".format(prediction_2, k))
            print()
    print()

    for k in k_values:
        valid_predictions_all_features, valid_predictions_exclude_age = 0, 0

        # test with leave-1-out training method
        for index, test_sample in df_dataset.iterrows():
            sample = test_sample.values[:3] # leave the target out
            target = test_sample.values[3]
            prediction = KNN_classification(sample, k, df_dataset.drop(index), drop_age=False)
            valid_predictions_all_features += 1 if target == prediction else 0
            # print("Prediction:{} - Target: {} for k: {} number of neighbors".format(prediction_1, target, k))

            prediction = KNN_classification(sample[:2], k, df_dataset.drop(index), drop_age=True) # assumption: gender is is the 3rd element of the sample
            valid_predictions_exclude_age += 1 if target == prediction else 0

            # print("Prediction: {} - Target: {} for k:{} number of neighbors without using age feature".format(prediction_2, target, k))

            # prediction = KNN_classification(sample[:2], k, df_dataset.drop(index),
            #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
            # valid_predictions_all_features += 1 if target == prediction else 0
        print("KNN Performance using k:{}".format(k))
        print("{}/{} correct predictions using all features".format(valid_predictions_all_features, df_dataset.shape[0]))
        print("{}/{} correct predictions excluding age".format(valid_predictions_exclude_age, df_dataset.shape[0]))
        print()


