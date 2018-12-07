import numpy as np
from numpy import log

def fromOneDimMatrixToArray( oneDimMatrix ):
    '''
    converts [1xn] matrix into a numpy array of shape( n, )

    returns a numpy array
    '''
    return np.array( oneDimMatrix )[ 0 ]

'''
Functions called by *classification.py
1. trainParameterGivenTopic( topicWordFrequencyArray, smoothingParam ):
    returns a numpy array of parameters, whose shape[0] = topicWordFrequencyArray.shape[0]

2. computeLogLikelihood( testWordFrequencyArray, parameterArray ):
    returns a scalar number
'''
# ---------------------------
# Model Specific Functions
def trainParameterGivenTopic( topicWordFrequencyArray, smoothingParam = 0 ):
    '''
    Given a numpy array where ith element corrseponds to freq of
    ith element in topic corpus, and a smoothing param
    returns a list of parameters that are ML estimates of prob occurence
    for each word

    returns an array of parameters:
    - size = len( topicWordFrequencyArray )
    - values are ML estimates of prob occurence for each word
    '''
    mlEstimateArray = topicWordFrequencyArray + smoothingParam
    denominator = np.sum( mlEstimateArray )
    return mlEstimateArray / denominator
def computeLogLikelihood( testWordFrequencyArray, parameterArray ):
    '''
    given a test sentence represented by an array of word-frequency
    and parameters for a multinomial distribution
    calcluate log-likelihood of seeing test sentence

    Assumes Naive Bayes assumption

    - testWordFrequencyArray: array of freqeucny count for word, numpy array
    - parameterArray: numpy array of size |lexicon|, corresponding to values of multinomial

    returns a number
    '''
    logParam = np.log( parameterArray )
    return np.dot( testWordFrequencyArray, logParam )


