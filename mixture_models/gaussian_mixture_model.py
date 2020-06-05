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

import math             # Single-value math operations
import numpy as np      # Vector math operations
from scipy.stats import multivariate_normal


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
    if cov_det == 0:
        print('Error, covariance matrix is singular')
        exit(-1)
        
    cov_inv = np.linalg.pinv(covariance_matrix)
    
    # Compute exponent of Gaussian distribution
    data_sub_means = data_array - means_array
    exponent = (-1/2) * np.dot( np.dot( np.reshape(data_sub_means, (1,len(data_array)) ), cov_inv),
                                        np.reshape(data_sub_means, (len(data_array),1) )) 
    
    # Compute probability of data array being sampled by the given Gaussian
    denominator = ((2 * math.pi) ** (len(data_array)/2)) * (cov_det ** (1/2))
    probability = (1 / denominator) * math.exp(exponent)
    
    return probability


# Data log likelihood for a mixture of Gaussians
def gaussian_mixture_log_likelihood(X_labeled, y_array, X_unlabeled, all_priors, all_means, all_covariances, num_classes):
    
    log_likelihood = 0.0
    # Labeled data likelihood
    for data_array, y in zip(X_labeled, y_array):
        class_conditional = multivariate_gaussian_pdf(data_array, all_means[y], all_covariances[y])            
        # log_likelihood += math.log( all_priors[y] * class_conditional )
        log_likelihood += all_priors[y] * class_conditional
    # Unlabeled data likelihood, marginalizing over the classes
    for data_array in X_unlabeled:
        class_marginalized_likelihood = 0.0
        for j in range(num_classes):
            class_conditional = multivariate_gaussian_pdf(data_array, all_means[j], all_covariances[j])
            class_marginalized_likelihood += all_priors[j] * class_conditional
        
        # log_likelihood += math.log ( class_marginalized_likelihood )
        log_likelihood += class_marginalized_likelihood
        
    return log_likelihood
    
    
# Given a data and weights matrices, return the MLE for a the Gaussian distribution
# Data matrix is (instances, dimensions)
# Weights matrix is (instances, num_classes)
def maximum_likelihood_estimation(X, weights, num_classes):
    if len(X) != len(weights):
        print("Error, data and weights matrices are incompatible")
        exit(-1)
        
    X = np.array(X)
    weights = np.array(weights)
    # Store means arrays and covariance matrices for all classes
    all_priors = []
    all_means = []
    all_covariances = []
    
    # Compute distribution parameters for each class
    for j in range(num_classes):
    
        class_len = np.sum(weights[:,j])
        # print(class_len)
        # Compute priors and means
        class_prior = class_len / len(weights)
        weighted_X = X * (weights[:, j])[:, None]
        class_means_array = np.sum(weighted_X, 0) / class_len
        dimensionality = len(class_means_array)
        
        # Compute covariance matrix
        # transp_class_X = (X[weights[:,j]==1]).T
        # np_cov = np.cov(transp_class_X, bias=True)
        # print(np_cov)
        
        class_cov_matrix = np.zeros((dimensionality,dimensionality))
        for i in range(len(weights)):
            if weights[i,j] != 0:
                data_minus_means = X[i] - class_means_array
                inner_prod = np.dot(np.reshape(data_minus_means, (dimensionality,1)), np.reshape(data_minus_means, (1,dimensionality)))
                submatrix = weights[i,j] * inner_prod
                class_cov_matrix += submatrix
                
        class_cov_matrix = class_cov_matrix / class_len  
        all_priors.append(class_prior)
        all_means.append(class_means_array)
        all_covariances.append(class_cov_matrix)
        
    return all_priors, all_means, all_covariances
    
# 
def expectation_maximization(X_labeled, X_unlabeled, y, num_iterations):
    num_classes = 2
    X_labeled = np.array(X_labeled)
    X_unlabeled = np.array(X_unlabeled)
    y = np.array(y)
    weights_labeled = np.transpose(np.vstack( (y, (1-y)) ))
    
    # Initialize MLE on labeled data
    # Priors are used for all instances
    all_priors, all_means, all_covariances = maximum_likelihood_estimation(X_labeled, weights_labeled, num_classes)
    
    ## Loop
    for _ in range(num_iterations):
        # Assign conditional probabilities p(y|x) to unlabeled data with Gaussian mixture
        unlabeled_conditionals = []
        for data_array in X_unlabeled:
            # Compute 
            array_conditionals = []
            for j in range(num_classes):
                class_conditional_prob = multivariate_gaussian_pdf(data_array, all_means[j], all_covariances[j])
                # print(multivariate_normal(all_means[j], all_covariances[j]).pdf(data_array))
                array_conditionals.append(all_priors[j] * class_conditional_prob)
            
            array_conditionals = np.array(array_conditionals)/sum(array_conditionals)
            unlabeled_conditionals.append(array_conditionals)
        
        # Compute MLE on whole data
        all_weights = np.vstack((weights_labeled, unlabeled_conditionals))
        all_priors, all_means, all_covariances = maximum_likelihood_estimation(np.vstack((X_labeled,X_unlabeled)), all_weights, num_classes)

        # Check for convergence
        mixture_log_likelihood = gaussian_mixture_log_likelihood(X_labeled, y, X_unlabeled, all_priors, all_means, all_covariances, num_classes)
        # print(mixture_log_likelihood)
    
    return mixture_log_likelihood, all_priors, all_means, all_covariances
    
    