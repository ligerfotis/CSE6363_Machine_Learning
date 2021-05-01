# from data import heights, weights, age, gender
from data import dataset_labeled, dataset_unlabeled, test_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
k_list = [1, 2, 3, 4, 5, 6]
# k = 4
for k in k_list:
    print("k:{}".format(k))
    x_data_labeled = np.array([sample[0] for sample in dataset_labeled])
    y_data = np.array([0 if sample[1] == 'M' else 1 for sample in dataset_labeled])

    x_data_labeled = pd.DataFrame(x_data_labeled, columns=['height', 'weight', 'age'])
    x_data_unlabeled = pd.DataFrame(dataset_unlabeled, columns=['height', 'weight', 'age'])

    x_test_labeled = np.array([sample[0] for sample in test_dataset])
    y_test = np.array([0 if sample[1] == 'M' else 1 for sample in test_dataset])

    # train a learner only on the initially labeled data
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_data_labeled, y_data)

    # semi-supervided training
    logisticRegr_semi = LogisticRegression()

    while len(x_data_unlabeled) > 0:
        if len(x_data_unlabeled) < k:
            k = len(x_data_unlabeled)

        logisticRegr_semi.fit(x_data_labeled, y_data)

        # unlabelled data confidence
        predictions = logisticRegr_semi.predict(x_data_unlabeled)
        confidence_2class = logisticRegr_semi.predict_proba(x_data_unlabeled)

        confidence = [con[int(pred)]for con, pred in zip(confidence_2class, predictions)]

        x_data_unlabeled_sorted = x_data_unlabeled.copy()
        x_data_unlabeled_sorted['predictions'] = predictions
        x_data_unlabeled_sorted['confidence'] = confidence

        x_data_unlabeled_sorted = x_data_unlabeled_sorted.sort_values(by='confidence', ascending=False)
        sorted_idx = list(x_data_unlabeled_sorted.index)
        # add newly labeled data to the labeled dataset
        for i in sorted_idx[:k]:
            x_data_labeled = x_data_labeled.append(x_data_unlabeled.iloc[i], ignore_index=True)
            y_data = np.concatenate([y_data, [x_data_unlabeled_sorted.iloc[i]['predictions']]])
            # remove recently labeled sample from the unlabeled dataset
        x_data_unlabeled = x_data_unlabeled.drop(sorted_idx[:k], axis=0)
        x_data_unlabeled.reset_index(drop=True, inplace=True)

    pred_simple = logisticRegr.predict(x_test_labeled)
    pred_semi = logisticRegr_semi.predict(x_test_labeled)

    accuracy_simple = (len(y_test) - np.count_nonzero(y_test - pred_simple)) / len(y_test)
    accuracy_semi = (len(y_test) - np.count_nonzero(y_test - pred_semi)) / len(y_test)

    print("Simple: {}% accuracy | Semi: {}% accuracy".format(accuracy_simple*100, accuracy_semi*100))

