from decision_trees import load_data, CART
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

train_data, test_data = load_data()

list_of_datasets = []
n_bagging = [10, 50, 100]


def bagging(bag_num):
    """
    Trains a classifier using bagging. It may contain duplicates.
    :param bag_num:
    :return: trained model (tree)
    """
    list_of_train_datasets = []
    for i in range(bag_num):
        indexes = np.random.randint(low=0, high=train_data.shape[0], size=train_data.shape[0])
        list_of_train_datasets.append(train_data.iloc[indexes])
    list_of_models = []
    for dataset in list_of_train_datasets:
        # create a CART instance
        cart = CART(max_depth=4)
        # train CART
        tree = cart.build_tree(dataset, 0, type="root", parent=None)
        list_of_models.append(cart)

    # testing dataset
    test_labels = test_data['label'].values
    x_test = test_data.loc[:, test_data.columns != 'label']
    count_correct_pred = 0
    for index, sample in x_test.iterrows():
        votes = []
        for model in list_of_models:
            vote = model.predict(sample, model.root)
            votes.append(vote)
        labels, counts = np.unique(votes, return_counts=True)
        if len(counts) == 1:
            prediction = labels
        else:
            prediction = labels[0] if counts[0] > counts[1] else labels[1]
        if prediction == test_labels[index]:
            count_correct_pred += 1
    accuracy = count_correct_pred / len(test_labels)
    return accuracy

#
# for bag_num in n_bagging:
#     accuracy = bagging(bag_num)
#     print("Bagging for {} times. Accuracy:{}".format(bag_num, accuracy))

