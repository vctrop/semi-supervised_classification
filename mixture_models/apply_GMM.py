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
from sklearn.datasets import load_iris

full_iris = load_iris()

# Restrict data to two features (petal len, wid)
y_array = np.array(full_iris.target)
iris_labels = y_array [ np.logical_or(y_array == 0, y_array == 1) ]
iris_data = full_iris.data [ np.logical_or(full_iris.target == 0, y_array == 1)]
# Restrict data to two classes (setosa, versicolor)
iris_data = iris_data[:, 0:2]

# Split data in labeled and unlabeled
labeled_data = iris_data[40:60]
partial_labels = iris_labels[40:60]
unlabeled_data = np.vstack((iris_data[:40], iris_data[60:]))

# print(np.shape(labeled_data))
# print(np.shape(partial_labels))
# print(np.shape(unlabeled_data))

log_likelihood, _, _, _ = gaussian_mixture_model.expectation_maximization(labeled_data, unlabeled_data, partial_labels, 1)
print(log_likelihood)