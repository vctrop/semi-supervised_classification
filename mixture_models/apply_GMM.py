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
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42)
rng = np.random.RandomState(2)
X += 3 * rng.uniform(size=X.shape)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
            s=25, edgecolor='k')
plt.plot()
plt.show()
            
            
print('[Fully supervised]')
weights_labeled = np.transpose(np.vstack( (y, (1-y)) ))
_, all_means, all_covariances = gaussian_mixture_model.maximum_likelihood_estimation(X, weights_labeled, 2)
print('f1 means: ' + str(all_means[0]))
print('f2 means: ' + str(all_means[1]))
print('f1 cov: \n' + str(all_covariances[0]))
print('f2 cov: \n' + str(all_covariances[1]))

# Split data in labeled and unlabeled
labeled_data = X[45:55]
partial_labels = y[45:55]
unlabeled_data = np.vstack((X[:45], X[55:]))

# print(np.shape(labeled_data))
# print(np.shape(partial_labels))
# print(np.shape(unlabeled_data))

iterations = [1, 100, 200]
for i in iterations:
    log_likelihood, _, all_means, all_covariances = gaussian_mixture_model.expectation_maximization(labeled_data, unlabeled_data, partial_labels, i)
    print('\n[Semi-supervised ' + str(i) + ' iteration]')
    print('f1 means: ' + str(all_means[0]))
    print('f2 means: ' + str(all_means[1]))
    print('f1 cov: \n' + str(all_covariances[0]))
    print('f2 cov: \n' + str(all_covariances[1]))


