import pandas as pd
import numpy as np
from math import e

BASE = 2


def load_data():
    data_train = pd.read_csv("tic-tac-toe_train.csv", header=None, names=["f" + str(i) for i in range(9)] + ["label"])
    data_test = pd.read_csv("tic-tac-toe_test.csv", header=None, names=["f" + str(i) for i in range(9)] + ["label"])
    return data_train, data_test


train_data, test_data = load_data()


def entropy(labels, base=None, boosting=False):
    """
    Taken from https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
    :param boosting:
    :param labels:
    :param base:
    :return:
    """
    if boosting:
        win_indexes = np.where(labels['label'] == 'win')
        no_win_indexes = np.where(labels['label'] == 'no-win')

        class_weights = np.asarray([labels.iloc[win_indexes]['weight'].sum(), labels.iloc[no_win_indexes]['weight'].sum()])
        prob = class_weights / class_weights.sum()
    else:
        value, counts = np.unique(labels, return_counts=True)
        prob = counts / counts.sum()
    base = e if base is None else base
    if 0 in prob:
        return 0
    return -(prob * np.log(prob) / np.log(base)).sum()


def information_gain(data, split_name, boosting=False):
    values, counts = np.unique(data[split_name], return_counts=True)

    # if not boosting and 'weight' in data.columns:
    #     data.drop('weight')

    label = ['weight', 'label'] if boosting else 'label'

    node_entropy = entropy(data[label], base=BASE, boosting=boosting)
    # do the split
    if 'x' in values:
        split1 = data[data[split_name] == "x"]
        split1_entropy = entropy(split1[label], base=BASE, boosting=boosting)
        weight = counts[np.where(values == 'x')] / sum(counts)
        weighted_entropy_split1 = weight * split1_entropy
    else:
        weighted_entropy_split1 = 0
    if 'o' in values:
        split2 = data[data[split_name] == "o"]
        split2_entropy = entropy(split2[label], base=BASE, boosting=boosting)
        weight = counts[np.where(values == 'o')] / sum(counts)
        weighted_entropy_split2 = weight * split2_entropy
    else:
        weighted_entropy_split2 = 0
    if 'b' in values:
        split3 = data[data[split_name] == "b"]
        split3_entropy = entropy(split3[label], base=BASE, boosting=boosting)
        weight = counts[np.where(values == 'b')] / sum(counts)
        weighted_entropy_split3 = weight * split3_entropy
    else:
        weighted_entropy_split3 = 0
    return node_entropy - (weighted_entropy_split1 + weighted_entropy_split2 + weighted_entropy_split3)


def best_split(data, boosting=False):
    gain_dict = {}
    best_feature, best_gain = None, None
    columns = data.columns.values
    if boosting and 'label' in columns:
        columns = np.delete(columns, np.where(columns == 'weight'))
    columns = np.delete(columns, np.where(columns == 'label'))

    for split_feature in columns:
        gain = information_gain(data=data, split_name=split_feature, boosting=boosting)
        gain_dict[split_feature] = gain
    # print(gain_dict)
    if len(gain_dict) > 0:
        best_feature = max(gain_dict, key=gain_dict.get)
        best_gain = gain_dict[best_feature][0]
    # print("Max Information Gain: {} at feature: {}".format(best_gain, best_feature))
    return best_feature, best_gain


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
        self.type = type  # 'x', 'o' or 'b'

    def __str__(self):
        return "{}_{}".format(self.split_feature, self.type)

    def calc_entropy(self, boosting):
        if boosting:
            self.entropy = entropy(self.df[['weight', 'label']], BASE, boosting=boosting)
        else:
            self.entropy = entropy(self.df['label'], BASE, boosting=boosting)


class CART:
    def __init__(self, max_depth, boosting):
        self.root = None
        self.max_depth = max_depth

    def build_tree(self, data_frame, depth, type, parent, boosting=False):
        if depth < 0:
            print("Depth cannot be a negative number")
            return False

        node = Node(parent_node=parent, data_frame=data_frame, type=type)

        # Split recursively until maximum depth is reached.
        if depth <= self.max_depth:

            node.calc_entropy(boosting)

            # node is a leaf
            if node.entropy == 0 or node.df.shape[0] == 1:
                return node

            split_feature, split_gain = best_split(data_frame, boosting)
            node.split_feature = split_feature

            if parent is not None:
                node.parent_node = parent
            else:
                node.name = type

            # print("Entropy of node {} is: {}".format(node.parent_node.split_feature + node.type, node.entropy))

            if split_feature is not None:
                # print("Parent: {} | Node: {} | Depth: {} | Split Feature {}".format(node.parent_node, node, depth,
                #                                                                     split_feature))
                node.split_data_o = data_frame[data_frame[split_feature] == 'o'].drop([split_feature], axis=1)
                node.split_data_b = data_frame[data_frame[split_feature] == 'b'].drop([split_feature], axis=1)
                node.split_data_x = data_frame[data_frame[split_feature] == 'x'].drop([split_feature], axis=1)

                # node.split_feature = split_feature
                if node.split_data_o.shape[0] != 0:
                    node.child_o = self.build_tree(node.split_data_o, depth + 1, type='o', parent=node, boosting=boosting)
                if node.split_data_x.shape[0] != 0:
                    node.child_x = self.build_tree(node.split_data_x, depth + 1, type='x', parent=node, boosting=boosting)
                if node.split_data_b.shape[0] != 0:
                    node.child_b = self.build_tree(node.split_data_b, depth + 1, type='b', parent=node, boosting=boosting)

            self.root = node
            return node

        # max depth reached; make the node a leaf
        else:
            return node

    def predict(self, sample, tree):
        if tree is None:
            print('here')
        if tree.entropy == 0:
            return self.classify(tree)

        if sample[tree.split_feature] == 'o':
            if tree.child_o is None:
                return self.classify(tree)
            else:
                return self.predict(sample, tree.child_o)
        elif sample[tree.split_feature] == 'x':
            if tree.child_x is None:
                return self.classify(tree)
            else:
                return self.predict(sample, tree.child_x)
        else:
            if tree.child_b is None:
                return self.classify(tree)
            else:
                return self.predict(sample, tree.child_b)

    def evaluate(self, data, labels):
        count_correct_pred = 0
        # prediction accuracy on training data
        for index, sample in data.iterrows():
            prediction = self.predict(sample, self.root)
            if prediction == labels[index]:
                count_correct_pred += 1
        accuracy = count_correct_pred / len(labels)
        return accuracy

    def classify(self, tree):
        value, counts = np.unique(tree.df['label'], return_counts=True)
        if len(value) == 2:
            class_pred = value[0] if counts[0] >= counts[1] else value[1]
            return class_pred
        # only one class left in the labels
        else:
            return value


#
# for max_depth in range(1, 10):
#     # create a CART instance
#     cart = CART(max_depth, boosting=False)
#     # train CART
#     tree = cart.build_tree(train_data, 0, type="root", parent=Node())
#     # train dataset
#     train_labels = train_data['label'].values
#     x_train = train_data.loc[:, train_data.columns != 'label']
#     # testing dataset
#     test_labels = test_data['label'].values
#     x_test = test_data.loc[:, test_data.columns != 'label']
#
#     train_accuracy = cart.evaluate(x_train, train_labels)
#
#     test_accuracy = cart.evaluate(x_test, test_labels)
#
#     print("Prediction accuracy for max depth {} is {:.2f}% on train data and {:.2f}% in testing data".format(max_depth, train_accuracy * 100, test_accuracy * 100))
