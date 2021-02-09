import numpy as np
import math

data = [2, 5, 0, 3, 1, 3]

alpha = 2
beta = 1


def poisson_sample(l, k):
    """
    Calculates the probability of k events occurring in a fixed interval l (lambda)
    :param l: lambda; fixed interval
    :param k: number of events
    :return: the probability of k occurring in l
    """
    return np.power(l, k) * math.exp(-l) * 1 / math.factorial(k)


# def gamma_sample(l, alpha, beta, k):
#     """
#     Poisson conjugate prior
#
#     Calculates the probability of k events occurring in a fixed interval l (lambda)
#     :param l: lambda; fixed interval
#     :param alpha:
#     :param beta:
#     :param k: number of events
#     :return: the probability of k occurring in l
#     """
#     return (np.power(beta, alpha) / math.gamma(alpha)) * np.power(l, alpha - 1) * math.exp(-beta * l)


def optimize_poisson_mle(dataset):
    """
    Performance Metric: MLE
    Optimizes the parameter lambda of a Poisson Distribution that the data follow
    l' = argmax_l( P(D|l)) = ... = (1/dataset_length)* Sum(dataset_elements) = mean(dataset)
    :param dataset: the dataset
    :return: optimized lambda
    """
    MLE_l = np.mean(dataset)
    return MLE_l


def optimize_poisson_map(dataset, alpha, beta):
    """
    Performance Metric: MAP
    Optimizes the parameter lambda of a Poisson Distribution that the data follow
    l' = argmax_l( P(l|D) * P(D)) = ... = (1/(dataset_length + beta))* (Sum(dataset_elements) + a)
    :param dataset: the dataset
    :return: optimized lambda
    """
    MAP_l = (1 / (len(dataset) + beta)) * (np.sum(dataset) + alpha)
    return MAP_l


if __name__ == '__main__':
    # optimize lambda parameter for MLE performance metric
    l_opt_mle = optimize_poisson_mle(data)

    # optimize lambda parameter for MLE performance metric
    l_opt_map = optimize_poisson_map(data, alpha=alpha, beta=beta)

    for sample in data:
        prediction_mle = poisson_sample(l_opt_mle, sample)
        prediction_map = poisson_sample(l_opt_map, sample)
        print("Probability of sample {} is: {} using MLE metric".format(sample, prediction_mle))
        print("Probability of sample {} is: {} using MAP metric".format(sample, prediction_map))
