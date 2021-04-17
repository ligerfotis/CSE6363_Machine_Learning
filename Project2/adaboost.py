from decision_trees import load_data, CART
import numpy as np

train_data, test_data = load_data()

n_bagging = [1, 15, 20, 50]

for iterations in n_bagging:
    # add a weight column at the training data
    train_data_weighted = train_data.copy()
    weights = np.ones(len(train_data)) / len(train_data)
    train_data_weighted['weight'] = weights

    models_list = []
    alpha_list = []
    for i in range(iterations):
        # create a CART instance
        cart = CART(max_depth=3, boosting=True)
        # train CART
        # root = cart.build_tree(train_data, 0, type="root", parent=None)
        root = cart.build_tree(train_data_weighted, 0, type="root", parent=None, boosting=True)
        test = test_data.loc[:, test_data.columns != 'weight']
        # print(cart.evaluate(test_data.loc[:, test_data.columns != 'label'], test_data['label'].values))
        train_labels = train_data_weighted['label'].values
        x_train = train_data_weighted.loc[:, train_data_weighted.columns != 'label']
        x_train = x_train.loc[:, x_train.columns != 'weight']

        error_list, prediction_list = [], []
        for index, sample in x_train.iterrows():
            prediction = cart.predict(sample, root)
            prediction_list.append(prediction[0])
            error_list.append(1 if prediction != train_labels[index] else 0)

        w = train_data_weighted['weight'].values
        epsilon = sum(w * error_list) / w.sum()

        if 0 < epsilon < 0.5:
            # keep this model
            models_list.append(cart)

            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            alpha_list.append(alpha)
            y = [1 if label == 'win' else -1 for label in train_labels]
            h = [1 if pred == 'win' else -1 for pred in prediction_list]
            weights = weights * np.exp(-alpha * np.asarray(y) * np.asarray(h))
            train_data_weighted['weight'] = weights

    # testing dataset
    test_labels = test_data['label'].values
    test_labels[test_labels == 'win'] = 1
    test_labels[test_labels == 'no-win'] = -1

    x_test = test_data.loc[:, test_data.columns != 'label']
    count_correct_pred = 0

    for index, sample in x_test.iterrows():
        # classify
        classifications = [model.predict(sample, model.root) for model in models_list]
        classifications_int = [1 if classification == 'win' else -1 for classification in classifications]
        prediction = np.sign(sum(alpha_list * np.asarray(classifications_int)))
        if prediction == test_labels[index]:
            count_correct_pred += 1

    accuracy = count_correct_pred / len(test_labels)
    print("Boosting for {} times. Accuracy: {}%".format(iterations, accuracy * 100))
