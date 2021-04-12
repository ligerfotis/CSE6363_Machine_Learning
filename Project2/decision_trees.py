import pandas as pd
import numpy as np
from math import log, e


def load_data():
    data_train = pd.read_csv("tic-tac-toe_train.csv", header=None, names=["f" + str(i) for i in range(9)] + ["label"])
    data_test = pd.read_csv("tic-tac-toe_test.csv", header=None, names=["f" + str(i) for i in range(9)] + ["label"])
    return data_train, data_test


x_train, x_test = load_data()


def entropy(labels, base=None):
    """
    Taken from https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
    :param labels:
    :param base:
    :return:
    """
    value, counts = np.unique(labels, return_counts=True)
    prob = counts / counts.sum()
    base = e if base is None else base

    return -(prob * np.log(prob) / np.log(base)).sum()


def feature_entropy(feature):
    # print(x_train[x_train[feature] == "x"]["label"])
    value1_feature = x_train[x_train[feature] == "x"]["label"]
    value2_feature = x_train[x_train[feature] == "o"]["label"]
    value3_feature = x_train[x_train[feature] == "b"]["label"]
    #
    # print(entropy(value1_feature))
    # print(entropy(value2_feature))
    # print(entropy(value3_feature))
    return entropy(value1_feature), entropy(value2_feature), entropy(value3_feature)


def information_gain(data, split_name, target_name):
    target_entropy = entropy(x_test[target_name], base=2)

    value, counts = np.unique(data[target_name], return_counts=True)
    prob = counts / counts.sum()
    # do the split
    split1 = data[data[split_name] == "x"][target_name]
    split2 = data[data[split_name] == "o"][target_name]
    split3 = data[data[split_name] == "b"][target_name]
    # calc the entropy of the split
    split1_entropy = entropy(split1, base=2)
    split2_entropy = entropy(split2, base=2)
    split3_entropy = entropy(split3, base=2)
    second_term = sum(prob * [split1_entropy, split2_entropy, split3_entropy])
    return target_entropy - second_term


def find_best_split(x_train):
    gains_names, gain_values = {}, {}
    for target_name in x_train.columns[:-1]:
        subgain = {}
        for split_name in x_train.columns[:-1]:
            if target_name != split_name:
                gain = information_gain(data=x_train, split_name=split_name, target_name=target_name)
                subgain[split_name] = gain
        max_gain_feature = max(subgain, key=subgain.get)
        gains_names[target_name] = max_gain_feature
        gain_values[target_name] = subgain[max_gain_feature]
    return gains_names, gain_values


names, values = find_best_split(x_train)
print(names)
print(values)
