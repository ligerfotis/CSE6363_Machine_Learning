import pandas as pd
import numpy as np
from math import log, e

max_depth = 2
BASE = 2


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


def information_gain(data, split_name):
    values, counts = np.unique(data[split_name], return_counts=True)

    node_entropy = entropy(data['label'], base=BASE)
    # do the split
    if 'x' in values:
        split1 = data[data[split_name] == "x"]
        split1_entropy = entropy(split1['label'], base=BASE)
        weight = counts[np.where(values == 'x')]/sum(counts)
        weighted_entropy_split1 = weight * split1_entropy
    else:
        weighted_entropy_split1 = 0
    if 'o' in values:
        split2 = data[data[split_name] == "o"]
        split2_entropy = entropy(split2['label'], base=BASE)
        weight = counts[np.where(values == 'o')]/sum(counts)
        weighted_entropy_split2 = weight * split2_entropy
    else:
        weighted_entropy_split2 = 0
    if 'b' in values:
        split3 = data[data[split_name] == "b"]
        split3_entropy = entropy(split3['label'], base=BASE)
        weight = counts[np.where(values == 'b')] / sum(counts)
        weighted_entropy_split3 = weight * split3_entropy
    else:
        weighted_entropy_split3 = 0
    return node_entropy - (weighted_entropy_split1 + weighted_entropy_split2 + weighted_entropy_split3)


def best_split(data):
    gain_dict = {}
    best_feature, best_gain = None, None
    for split_feature in data.columns[:-1]:
        gain = information_gain(data=data, split_name=split_feature)
        gain_dict[split_feature] = gain

    if len(gain_dict) > 0:
        best_feature = max(gain_dict, key=gain_dict.get)
        best_gain = gain_dict[best_feature]
    print("Max Information Gain: {} at feature: {}".format(best_gain, best_feature))
    return best_feature, best_gain


"""
GAIN RATIO?
How to do the split?
"""


class Node:
    # https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
    def __init__(self, parent_node=None, data_frame=None, type=None):
        self.split_feature = "Leaf"
        self.child_x = None
        self.child_o = None
        self.child_b = None
        self.parent_node = parent_node
        self.name = self.parent_node
        self.entropy = 0
        self.df = data_frame
        self.type = type    # 'x', 'o' or 'b'

    def __str__(self):
        return "{}_{}".format(self.split_feature, self.type)

    def calc_entropy(self):
        self.entropy = entropy(self.df['label'], BASE)


class CART:
    def __init__(self):
        pass

    def build_tree(self, data_frame, depth, type, parent):
        if depth < 0:
            print("Depth cannot be a negative number")
            return False

        node = Node(parent_node=parent, data_frame=data_frame, type=type)

        # Split recursively until maximum depth is reached.
        if depth <= max_depth:

            node.calc_entropy()
            print("Entropy of node is: {}".format(node.entropy))

            # node is a leaf
            if node.entropy == 0:
                return node

            split_feature, split_gain = best_split(data_frame)
            node.split_feature = split_feature

            if parent is not None:
                node.parent_node = parent
            else:
                node.name = type

            if split_feature is not None:
                print("Parent: {} | Node: {} | Depth: {} | Split Feature {}".format(node.parent_node, node, depth,
                                                                                    split_feature))
                node.split_data_o = data_frame[data_frame[split_feature] == 'o'].drop([split_feature], axis=1)
                node.split_data_b = data_frame[data_frame[split_feature] == 'b'].drop([split_feature], axis=1)
                node.split_data_x = data_frame[data_frame[split_feature] == 'x'].drop([split_feature], axis=1)

                # node.split_feature = split_feature
                node.child_o = self.build_tree(node.split_data_o, depth + 1, type='o', parent=node)
                node.child_x = self.build_tree(node.split_data_x, depth + 1, type='x', parent=node)
                node.child_b = self.build_tree(node.split_data_b, depth + 1, type='b', parent=node)

        # max depth reached; make the node a leaf
        else:
            return node

    def predict(self, sample, tree):
        if tree.name is None:
            return


#
cart = CART()
tree = cart.build_tree(x_train, 0, type="root", parent=None)
# predict(x_test, tree)
#
# column_names = x_train.columns
# data_x = x_train[column_names[:-1]]
# labels = x_train[column_names[-1]]
#
# sample = data_x.iloc[0]
# predict(sample, tree)

# for index, sample in data_x.iterrows():
# print(sample.values)
