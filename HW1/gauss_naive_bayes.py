import math

import numpy as np
import pandas as pd

from data import heights, weights, age, gender, samples


def calculate_gaussian_probability(sample, mu, sigma):
    return 1 / (math.sqrt(sigma ** math.pi)) * np.exp(-sigma * np.power((sample - mu), 2))


def pdf_calculate(sample, feature, df_dataset):
    """
    Calculates the Probability Density Function (PDF) of 2 classes; 'M' and 'W' here.
    :param feature: feature to calulate PDF for
    :return: probability for each class
    """
    p_feature_men_mean = np.mean(df_dataset.loc[df_dataset['gender'] == 'M'][feature].values)
    p_feature_men_std = np.std(df_dataset.loc[df_dataset['gender'] == 'M'][feature].values)
    pdf_feature_men = calculate_gaussian_probability(sample, p_feature_men_mean, p_feature_men_std)

    p_feature_women_mean = np.mean(df_dataset.loc[df_dataset['gender'] == 'W'][feature].values)
    p_feature_women_std = np.std(df_dataset.loc[df_dataset['gender'] == 'W'][feature].values)
    pdf_feature_women = calculate_gaussian_probability(sample, p_feature_women_mean, p_feature_women_std)

    return pdf_feature_men, pdf_feature_women


def gaussian_naive_bayes_classification(sample, df_dataset, drop_age):
    """
    Naive Assumption -> every feature is independent from each other
    Thus, P(height, weight, age | class_i) = P(height| class_i)*P(weight| class_i)*P(age| class_i)
    Two classes: "M" & "W"
    :param sample:
    :param df_dataset:
    :return: predicted class
    """
    # Calculate PDFs for each feature
    pdf_height_men, pdf_height_women = pdf_calculate(sample[0], 'heights', df_dataset)
    pdf_weight_men, pdf_weight_women = pdf_calculate(sample[1], 'weights', df_dataset)

    # calculate prior probabilities of the classes
    num_of_men, num_of_women = np.count_nonzero(np.asarray(gender) == 'M'), np.count_nonzero(np.asarray(gender) == 'W')
    total_num_of_classes = num_of_women + num_of_men
    prior_men, prior_women = num_of_men / total_num_of_classes, num_of_women / total_num_of_classes

    if drop_age:
        # P(Class|Data) = P(Data|Class) * P(Class)
        # "Naive" -> P(feature_1, feature_2| class) = P(feature_1|Class)*P(feature_2|Class)
        p_man = pdf_height_men * pdf_weight_men * prior_men
        p_woman = pdf_height_women * pdf_weight_women * prior_women
    else:
        pdf_age_men, pdf_age_women = pdf_calculate(sample[2], 'age', df_dataset)

        # P(Class|Data) = P(Data|Class) * P(Class)
        # "Naive" -> P(feature_1, feature_2, feature_3| class) = P(feature_1|Class)*P(feature_2|Class)*P(feature_3|Class)
        p_man = pdf_height_men * pdf_weight_men * pdf_age_men * prior_men
        p_woman = pdf_height_women * pdf_weight_women * pdf_age_women * prior_women

    return 'M' if p_man > p_woman else 'W'


if __name__ == '__main__':

    df_dataset = pd.DataFrame({'heights': heights, 'weights': weights, 'age': age, 'gender': gender})

    for sample in samples:
        print("sample:{}".format(sample))
        prediction_1 = gaussian_naive_bayes_classification(sample, df_dataset, drop_age=False)
        print("\tPrediction is {}".format(prediction_1))
        # prediction_2 = KNN_classification(sample[:2], k, df_dataset,
        #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
        # print("\tPrediction is {} for k:{} number of neighbors without using age feature".format(prediction_2, k))
        print()


    valid_predictions_all_features, valid_predictions_exclude_age = 0, 0

    # test with leave-1-out training method
    for index, test_sample in df_dataset.iterrows():
        sample = test_sample.values[:3]  # leave the target out
        target = test_sample.values[3]
        prediction = gaussian_naive_bayes_classification(sample, df_dataset.drop(index), drop_age=False)
        valid_predictions_all_features += 1 if target == prediction else 0
        # print("Prediction:{} - Target: {} for k: {} number of neighbors".format(prediction_1, target, k))

        prediction = gaussian_naive_bayes_classification(sample[:2], df_dataset.drop(index),
                                                         drop_age=True)  # assumption: gender is is the 3rd element of the sample
        valid_predictions_exclude_age += 1 if target == prediction else 0
        # print("Prediction: {} - Target: {} for k:{} number of neighbors without using age feature".format(prediction_2, target, k))

        # prediction = KNN_classification(sample[:2], k, df_dataset.drop(index),
        #                                   drop_age=True)  # assumption: gender is is the 3rd element of the sample
        # valid_predictions_all_features += 1 if target == prediction else 0
    print("Gaussian Naive Performance")
    print("{}/{} correct predictions using all features".format(valid_predictions_all_features, df_dataset.shape[0]))
    print("{}/{} correct predictions excluding age".format(valid_predictions_exclude_age, df_dataset.shape[0]))
    print()
