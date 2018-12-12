import pickle
from scipy.sparse import csr_matrix # for fetching rows
import numpy as np
from collections import Counter
import os

def getFilename( splitNumber, isTraining, isLabel,
                 dataDirName = 'training_testing_and_results_pickles',
                 testCluster = 'twenty_newsgroups',
                 extensionName = 'pickle',
                 isUsingMatlab = False
                 ):
    '''
    returns filename to load from already test-train-split data
    4 mandatory options:
    - splitNumber: integer
    - isTraining: True if file for training
    - isLabel: True if file for getting label associated to matrix
    - isUsingMAtlabl: True if used to import learnt parameters from matlab

    SIDE effect: if directory doesn't exist, directory will be created

    returns a string with prefixes attace
    '''
    # recent git update renamed directories for consistent naming scheme
    alternateClusterName = testCluster
    trainingTestingStr = 'training' if isTraining else 'testing'
    matrixLabelStr = '_labels_vector_' if isLabel else '_'
    trainingTestingMatrixStr = 'trained_param' if isUsingMatlab else trainingTestingStr + '_matrix' 
    fName = alternateClusterName + '_' + trainingTestingMatrixStr + \
            matrixLabelStr + 'cv#' + str( splitNumber ) + '.' + extensionName
    prefix = './'+ dataDirName + '/' + testCluster + '/cv' + str( splitNumber ) + '/'
    os.makedirs( prefix, exist_ok=True )
    return prefix + fName

def partitionDataPerTopic( dataTopics, dataM ):
    '''
    Given a list of topics (asscoiated with each row in dataM) and dataM,
    return a list of unique topics, and a dictionary whose
    key: label, value: submatrix whose rows have value label
    (i.e. dataTopics[i] = label -> dataM.getrow(i) is a row in d[label])

    - dataTopics: a list of topics associated with each row
                    o has a special structure: [ 0,0,0,.... ,0,1,1,...,1,2...,2.. and so on]
    - dataM: a matrix with efficient slicing and row reading
    
    returns a list of unique topics, and a dictionary described above
    '''
    topicToMatrixD = {}
    topicCounts = Counter( dataTopics )
    startIdx = 0
    # UPDATE variables: startIdx, endIdx
    for topic in topicCounts.keys():
        endIdx = startIdx + topicCounts[ topic ]
        topicToMatrixD[ topic ] = dataM[ startIdx:endIdx ]
        startIdx = endIdx
    return (list( topicCounts.keys() ), topicToMatrixD)

if __name__ == '__main__':
    '''
    Step 1. Read and load training/test data split
    Step 2. partition training/test data per topic
    '''
    # --------------------------------
    # Step 1. Read and load training/test data split
    splitNumber = 0
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False ) 
    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True ) 
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False ) 
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True ) 

    #garboF = getFilename( splitNumber, False, False, dataDirName='foo', alternateClusterName='')
    with open( trainingMatF, 'rb' ) as f:
        trainingMat = csr_matrix( pickle.load( f ) )
        # fast matrix, whose get_row returns ONLY non-zero values
    with open( trainingLabelF, 'rb' ) as f:
        trainingLabel = pickle.load( f )
        # trainingLabel is a list
        # ordered such that 0 appears x_0, 1 appears x_1, and so forth
    with open( testingMatF, 'rb' ) as f:
        testingMat = csr_matrix( pickle.load( f ) )
    with open( testingLabelF, 'rb' ) as f:
        testingLabel = pickle.load( f )
    # --------------------------------
    # Step 2. partition training/test data per topic 
 
    (topicList, partitionedTrainingD) = partitionDataPerTopic( trainingLabel, trainingMat )
    (_, partitionedTestingD) = partitionDataPerTopic( testingLabel, testingMat )
    # key: topic, val: matrix whose col dimensions = |lexicon|, row dim = # docs belonging to topic

    trainingWordFrequencyD = {}
    # key: topic, value: array of topic wide frequency, len( array ) = | lexicon |
    for topic in topicList:
        # sum frequency row-wise
        topicFrequencyMat = partitionedTrainingD[ topic ].sum( axis = 0 )
        trainingWordFrequencyD[ topic ] = np.array( topicFrequencyMat )[0]
