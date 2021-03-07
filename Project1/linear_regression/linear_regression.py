import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

max_degree = 4

"""""""""""
Question a
"""""""""""


def load_train_data(filename):
    dataset = np.loadtxt(filename)
    feature_1, feature_2, labels = [item for item in dataset.T]
    data = np.asarray([feature_1, feature_2]).T
    labels = np.asarray(labels).T
    return data, labels


def polynomial_feature(input, degree):
    """
    Calculates Φ
    :param input: dataset {samples x features}
    :param degree: degree of polynomial
    :return: Φ {samples x features x polynomial degree}
    """
    assert input.shape[1] == 2
    features = []
    # polynomials
    for tmp_order in range(degree + 1):
        k1, k2 = tmp_order, 0
        while k1 >= 0:
            features.append(np.power(input[:, 0], k1) * np.power(input[:, 1], k2))
            k1 += -1
            k2 += 1

    return np.asarray(features).T


def fit_analytic(feature, labels):
    """
    calculates the analytic solution for linear regression
    :param featues:
    :param labels:
    :return: theta
    """
    pseudo_inverse = np.linalg.pinv(feature)
    theta = np.dot(pseudo_inverse, labels)
    return theta


def predict(features, theta_parameters):
    return np.dot(features, theta_parameters)


def create_input_meshgrid(data):
    # create a meshgrid for the input data space
    x1 = np.sort(data[:, 0])
    x2 = np.sort(data[:, 1])
    X1, X2 = np.meshgrid(x1, x2)

    # flatten the features tables into a 1D array
    mesh_data = np.vstack([X1.ravel(), X2.ravel()]).T

    return mesh_data, X1, X2


# load training data; input and train_labels
train_input, train_labels = load_train_data("data/PolyTrain.txt")

"""""""""""
Question b
"""""""""""
# create a meshgrid for the input data space
mesh_data, X1, X2 = create_input_meshgrid(train_input)

for degree in range(1, max_degree + 1):
    # get the polynomial feature matrix
    pol_features = polynomial_feature(train_input, degree)

    # fit the data to the linear regression model
    theta = fit_analytic(pol_features, train_labels)

    # plot a figure containing the models plane and the dataset points
    fig = plt.figure("Polynomial Regression for Degree {}".format(degree))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Polynomial Regression for Degree {}".format(degree))

    # get the polynomial feature table for the meshgrids
    pol_features_meshgrid = polynomial_feature(mesh_data, degree)

    # get the plane from based on the polynomial features of the meshgrid input points
    y = predict(pol_features_meshgrid, theta)
    # reshape the 1D y into 2D meshgrid predictions
    Y = np.reshape(y, X1.shape)

    # plot the 3D contour of the plane
    ax.contour3D(X1, X2, Y, 100, cmap='coolwarm')
    # add the data points on the plot
    ax.scatter(train_input[:, 0], train_input[:, 1], train_labels, label='Data', color='r')
    # change the view of the 3D plot
    ax.view_init(10, 30)
    ax.set_xlabel('$X1$')
    ax.set_ylabel('$X2$')
    ax.set_zlabel('$Y$')
    # save the figure
    # plt.show()
    plt.savefig("linear_regression/{}_order_poly_regression.png".format(degree))

"""""""""""
Question c
"""""""""""
# load test data; input and train_labels
test_input, test_labels = load_train_data("data/PolyTest.txt")

for degree in range(1, max_degree + 1):
    # get the polynomial feature matrix
    pol_features_train = polynomial_feature(train_input, degree)
    # fit the data to the linear regression model
    theta = fit_analytic(pol_features_train, train_labels)

    # calculate feature matrix for test data
    pol_features_test = polynomial_feature(test_input, degree)

    # calculate predictions
    predictions = predict(pol_features_test, theta)

    squared_error = (np.square(predictions - test_labels)).mean(axis=0)
    print("MSE: {} for degree {}".format(round(squared_error, 4), degree))
