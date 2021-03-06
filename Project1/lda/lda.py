from hw1_data import heights, weights, age, gender, samples
import numpy as np
from lda.utils import plot_data

dataset = np.asarray([heights, weights, age]).T
labels = np.asarray(gender)
test_data = np.asarray(samples)

"""""""""""
Question a
"""""""""""


def lda_predict(w, mu_list, sample):
    mu_0, mu_1 = mu_list

    criterion = w.dot(0.5 * (mu_0 + mu_1))
    if w.dot(sample) > criterion:
        return "M"
    else:
        return "W"


def lda_fit(dataset, labels):
    classes = np.unique(labels)
    n_features = dataset.shape[1]
    cov_matrix = np.zeros((n_features, n_features))
    mu_list = []
    for c in classes:
        x_c = dataset[labels == c]
        mu_c = np.mean(x_c, axis=0)
        mu_list.append(mu_c)
        cov_matrix += np.dot((x_c - mu_c).T, (x_c - mu_c)) / len(x_c)

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    w = (mu_list[0] - mu_list[1]).dot(inv_cov_matrix)

    return w, mu_list, cov_matrix


w, mu_list, cov = lda_fit(dataset, labels)
mu_0, mu_1 = mu_list
"""""""""""
Question b
"""""""""""
for test_sample in test_data:
    prediction = lda_predict(w, mu_list, test_sample)
    print(prediction)

"""""""""""
Question c
"""""""""""
sim_data_0, sim_data_1 = [], []

for i in range(50):
    sim = np.random.multivariate_normal(mu_0, cov)
    sim_data_0.append(sim)

    sim = np.random.multivariate_normal(mu_1, cov)
    sim_data_1.append(sim)

sim_data_0, sim_data_1 = np.asarray(sim_data_0), np.asarray(sim_data_1)

plot_data(dataset, sim_data_0, sim_data_1, labels)