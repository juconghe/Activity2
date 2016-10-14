# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.

As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).

"""

import numpy as np

def extractColumn(column,window):
    return window[:column]

def compute_crossing(crossingValue,valueArray):
    crossingCounter = 0
    for i in range(len(valueArray)-1):
        if((valueArray[i]>crossingValue && valueArray[i+1]<crossingValue)
        || (valueArray[i]<crossingValue && valueArray[i+1]>crossingValue)):
            crossingCounter += 1
    return crossingCounter

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)

def _compute_variance_features(window):
    return np.var(window,axis=0)

def _compute_min_features(window):
    return np.amin(window,axis=0)

def _compute_max_features(window):
    return np.amax(window,axis=0)

def _compute_median_features(window):
    return np.median(window,axis=0)

def _compute_zero_crossing_features(window):
    xArray = extractColumn(0,window)
    yArray = extractColumn(1,window)
    zArray = extractColumn(2,window)
    xCrossingRate = compute_crossing(0,xArray)
    yCrossingRate = compute_crossing(0,yArray)
    zCrossingRate = compute_crossing(0,zArray)
    return np.array([xCrossingRate,yCrossingRate,zCrossingRate])

def _compute_mean_crossing_features(window):
    meanArray = _compute_mean_features(window)
    xArray = extractColumn(0,window)
    yArray = extractColumn(1,window)
    zArray = extractColumn(2,window)
    xCrossingRate = compute_crossing(meanArray[0],xArray)
    yCrossingRate = compute_crossing(meanArray[1],yArray)
    zCrossingRate = compute_crossing(meanArray[2],zArray)
    return np.array([xCrossingRate,yCrossingRate,zCrossingRate])

def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature matrix X.

    Make sure that X is an N x d matrix, where N is the number
    of data points and d is the number of features.

    """

    x = []

    x = np.append(x, _compute_mean_features(window))
    x = np.append(x,_compute_variance_features(window))
    x = np.append(x,_compute_min_features(window))
    x = np.append(x,_compute_max_features(window))
    x = np.append(x,_compute_median_features(window))
    x = np.append(x,_compute_zero_crossing_features(window))
    x = np.append(x,_compute_mean_crossing_features(window))

    return x
