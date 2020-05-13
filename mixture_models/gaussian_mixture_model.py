#!python3

# Copyright (C) 2020  Academic League of Neurosciences (NeuroLiga), from Federal University of Santa Maria (UFSM) 

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

import math             # Single-value math operations
import numpy as np      # Vector math operations

# Compute the probability of a given data array being sampled from a given multivariate Gaussian distribution
def multivariate_gaussian_pdf(data_array, means_array, covariance_matrix):
    data_array = np.array(data_array)
    means_array = np.array(means_array)
    covariance_matrix = np.array(covariance_matrix)
    
    if len(data_array) != len(means_array):
        print("Error, data and means arrays must have the same dimensionality")
        exit(-1)
    if len(covariance_matrix[:,0]) != len(covariance_matrix[0,:]) or len(covariance_matrix[:,0]) != len(data_array):
        print("Error, invalid dimensionality for covariance matrix")
        exit(-1)
    
    # Compute determinant and inverse of covariance matrix
    cov_det = np.linalg.det(covariance_matrix)
    cov_inv = np.linalg.inv(covariance_matrix)
    
    # Compute exponent of Gaussian distribution
    data_sub_means = data_array - means_array
    exponent = (-1/2) * np.dot( np.dot(np.reshape(data_sub_means, (1,len(data_array))), cov_inv),
                                np.reshape(data_sub_means(len(data_array),1))) 
    
    # Compute probability of data array being sampled by the given Gaussian
    denominator = ((2 * math.pi) ** (len(data_array)/2)) * (cov_det ** 2)
    probability = (1 / denominator) * math.exp(exponent)
    
    return probability

    
#  
def maximum_likelihood_estimation(X, y, num_classes):
    if len(X) != len(y):
        print("Error, data matrix and labels array are incompatible")
        exit(-1)
        
    X = np.array(X)
    y = np.array(y)
    
    # Store means arrays and covariance matrices for all classes
    all_priors = []
    all_means = []
    all_covariances = []
    
    # Compute distribution parameters for each class
    for j in range(num_classes):
        X_class_j = X[y == j]
        
        class_prior = np.sum((y == j) * 1) / len(y)
        class_means_array = np.sum(X_class_j, 0) / np.sum((y == j) * 1)
        
        class_cov_matrix  = np.cov(X_class_j, X_class_j, ddof = 1)
        
        all_priors.append(class_prior)
        all_means.append(class_means_array)
        all_covariances.append(class_cov_matrix)
    
    return all_priors, all_means, all_covariances

    
# 
def expectation_maximization(X_labeled, X_unlabeled, y, num_iterations):
    num_classes = 2
    # Initialize MLE on labeled data
    # Priors are used for all instances
    all_priors, all_means, all_covariances = maximum_likelihood_estimation(X_labeled, y)
    
    ## Loop
    for _ in range(num_iterations):
        # Assign conditional probabilities p(y|x) to unlabeled data with Gaussian mixture
        unlabeled_conditionals = []
        for data_array in X_unlabeled:
            # Compute 
            array_conditionals = []
            for j in range(num_classes):
                class_prior = all_priors[j]
                class_means_array = all_means[j]
                class_cov_matrix = all_covariances[j]
                
                class_conditional_prob = multivariate_gaussian_pdf(data_array, class_means_array, class_cov_matrix)
                array_conditionals.append(class_prior * class_conditional_prob)
            
            array_conditionals = np.array(array_conditionals)/np.sum(array_conditionals)
            unlabeled_conditionals.append(array_conditionals)
        
        # Compute MLE on whole data
    
    
    
    pass
    
    