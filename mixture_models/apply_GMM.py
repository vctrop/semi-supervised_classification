#!python3

# Copyright (C) 2020  Academic League of Neurosciences (NeuroLiga), from Federal University of Santa Maria (UFSM) 
# Available at <https://github.com/vctrop/semi-supervised_drowsiness_detection/>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#import math                    # Single-value math operations
import numpy as np              # Vector math operations
import gaussian_mixture_model   #
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# # IRIS DATASET
# full_iris = load_iris()
# # Restrict data to two features (petal len, wid)
# y_array = np.array(full_iris.target)
# iris_data = full_iris.data [ np.logical_or(full_iris.target == 0, y_array == 1)]
# # Restrict data to two classes (setosa, versicolor)
# iris_labels = y_array [ np.logical_or(y_array == 0, y_array == 1) ]
# iris_data = iris_data[:, 0:2]

# Synthetic Gaussian dataset
# Train data 
X_train, y_train = make_classification(n_samples=200, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42, class_sep=0.8 , shuffle=False)
rng = np.random.RandomState(2)
X_train += 2 * rng.uniform(size=X_train.shape)

# # Test data
# X_test, y_test = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X_test += 3 * rng.uniform(size=X_test.shape)

# Plot generated dataset
# plt.figure()
# plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train,
            # s=25, edgecolor='k')
# plt.plot()
# plt.show()

print('[Fully supervised]')
weights_labeled = np.transpose(np.vstack( ((1-y_train), y_train) ))
all_priors, all_means, all_covariances = gaussian_mixture_model.maximum_likelihood_estimation(X_train, weights_labeled, 2)
print('f1 means: ' + str(all_means[0]))
print('f2 means: ' + str(all_means[1]))
print('f1 cov: \n' + str(all_covariances[0]))
print('f2 cov: \n' + str(all_covariances[1]))

y_pred, predicted_probabilities = gaussian_mixture_model.predict(X_train, all_priors, all_means, all_covariances)
accuracy = accuracy_score(y_train, y_pred)
print('acc: ' + str(accuracy*100))
# print(predicted_probabilities)
# print(y_train)
# print(y_pred)

## SSL
# Split data in labeled and unlabeled
labeled_data = X_train[90:110]
partial_labels = y_train[90:110]
unlabeled_data = np.vstack((X_train[:90], X_train[110:]))
# print(np.shape(labeled_data))
# print(np.shape(partial_labels))
# print(np.shape(unlabeled_data))

print('\n[SSL: labeled data only]')
weights_labeled = np.transpose(np.vstack( ((1-partial_labels), partial_labels) ))
all_priors, all_means, all_covariances = gaussian_mixture_model.maximum_likelihood_estimation(labeled_data, weights_labeled, 2)
print('f1 means: ' + str(all_means[0]))
print('f2 means: ' + str(all_means[1]))
print('f1 cov: \n' + str(all_covariances[0]))
print('f2 cov: \n' + str(all_covariances[1]))

y_pred, predicted_probabilities = gaussian_mixture_model.predict(X_train, all_priors, all_means, all_covariances)
accuracy = accuracy_score(y_train, y_pred)
print('acc: ' + str(accuracy*100))

iterations = [1, 100, 1000]
for i in iterations:
    print('\n[SSL: EM iteration ' + str(i) + ']')
    log_likelihood, all_priors, all_means, all_covariances = gaussian_mixture_model.expectation_maximization(labeled_data, unlabeled_data, partial_labels, i)
    print('f1 means: ' + str(all_means[0]))
    print('f2 means: ' + str(all_means[1]))
    print('f1 cov: \n' + str(all_covariances[0]))
    print('f2 cov: \n' + str(all_covariances[1]))
    
    y_pred, predicted_probabilities = gaussian_mixture_model.predict(X_train, all_priors, all_means, all_covariances)
    accuracy = accuracy_score(y_train, y_pred)
    print('acc: ' + str(accuracy*100))
    

