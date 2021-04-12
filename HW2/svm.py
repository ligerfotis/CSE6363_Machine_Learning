from sklearn import svm, metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
"""
Code based on this article: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
"""
# Create a svm Classifier
from data import linear_dataset, linear_labels

clf = svm.SVC(C=1000, kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(linear_dataset, linear_labels)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# create a mesh to plot in
x_min, x_max = np.asarray(linear_dataset)[:, 0].min() - 1, np.asarray(linear_dataset)[:, 0].max() + 1
y_min, y_max = np.asarray(linear_dataset)[:, 1].min() - 1, np.asarray(linear_dataset)[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                     np.arange(y_min, y_max, .2))
Z = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])

Z = Z.reshape(xx2.shape)
plt.figure()
fig, ax = plt.subplots()
ax.contourf(xx2, yy2, Z, cmap=cm.coolwarm, alpha=0.3)
ax.scatter(np.asarray(linear_dataset)[:, 0], np.asarray(linear_dataset)[:, 1], c=linear_labels, cmap=cm.coolwarm, s=25)
ax.plot(xx, yy)

ax.axis([x_min, x_max,y_min, y_max])
plt.savefig("svm_plane.png")

# Predict the response for test dataset
y_pred = clf.predict(linear_dataset)

print(clf.support_vectors_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(linear_labels, y_pred))