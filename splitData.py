import pickle
from scipy.sparse import csr_matrix # for fetching rows
import numpy as np
from collections import Counter

def getFilename( splitNumber, isTraining, isLabel,
                 dataDirName = 'training_testing_and_results_pickles',
                 testCluster = '20_newsgroups',
                 alternateClusterName = 'twenty_newsgroups',
                 extensionName = 'pickle'
                 ):
    '''
    returns filename to load from already test-train-split data
    3 mandatory options:
    - splitNumber: integer
    - isTraining: True if file for training
    - isLabel: True if file for getting label associated to matrix

    returns a string with prefixes attached
    '''
    trainingTestingStr = 'training' if isTraining else 'testing'
    matrixLabelStr = '_labels_vector_' if isLabel else '_'
    fName = alternateClusterName + '_' + trainingTestingStr + '_matrix' + \
            matrixLabelStr + 'cv#' + str( splitNumber ) + '.' + extensionName
    prefix = './'+ dataDirName + '/' + testCluster + '/cv' + str( splitNumber ) + '/'
    return prefix + fName

def partitionTrainingDataPerTopic( trainingTopics, trainingDataM ):
    '''
    Given a list of topics (asscoiated with each row in trainingDataM) and trainingDataM,
    return a list of unique topics, and a dictionary whose
    key: label, value: submatrix whose rows have value label
    (i.e. trainingLabel[i] = label -> trainingDataM.getrow(i) is a row in d[label])

    - trainingLabel: a list of topics associated with each row
                    o has a special structure: [ 0,0,0,.... ,0,1,1,...,1,2...,2.. and so on]
    - trainingDataM: a matrix with efficient slicing and row reading
    
    returns a list of unique topics, and a dictionary described above
    '''
    topicToMatrixD = {}
    topicCounts = Counter( trainingTopics )
    startIdx = 0
    # UPDATE variables: startIdx, endIdx
    for topic in topicCounts.keys():
        endIdx = startIdx + topicCounts[ topic ]
        topicToMatrixD[ topic ] = trainingDataM[ startIdx:endIdx ]
        startIdx = endIdx
    return (topicCounts.keys(), topicToMatrixD)

if __name__ == '__main__':
    splitNumber = 0
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False ) 
    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True ) 
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False ) 
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True ) 

    with open( trainingMatF, 'rb' ) as f:
        trainingMat = csr_matrix( pickle.load( f ) )
        # fast matrix, whose get_row returns ONLY non-zero values
    with open( trainingLabelF, 'rb' ) as f:
        trainingLabel = pickle.load( f )
        # trainingLabel is a list
        # ordered such that 0 appears x_0, 1 appears x_1, and so forth
    
    #partitioning training data
    (topicList, partitionedTrainingD) = partitionTrainingDataPerTopic( trainingLabel, trainingMat )
    topicWordDictionary = {} 
    for topic in topicList:
        # sum frequency row-wise
        topicWordDictionary[ topic ] = partitionedTrainingD[ topic ].sum( axis = 0 )
    #print( trainingMat[0:23].sum( axis = 0 ) ) 
    #for key, value in trainingMat.items():
    #    print( key, value )
