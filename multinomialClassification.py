import pickle
import numpy as np
from numpy import log
import operator

#command line args
import sys

# Data splitting is done in another file
from scipy.sparse import csr_matrix
import splitData
from splitData import getFilename, partitionDataPerTopic

# reporting
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from statistics import mean, stdev
def fromOneDimMatrixToArray( oneDimMatrix ):
    '''
    converts [1xn] matrix into a numpy array of shape( n, )

    returns a numpy array
    '''
    return np.array( oneDimMatrix )[ 0 ]

# ---------------------------
# Model Specific Functions
import multinomialModel

# ----------------------------
# Generic function
def predict( testingMat, parameterArrayD, topicList, evalScoreF):
    '''
    Predict output topics of testingMat via evalScoreF x parameterArray

    - testingMat: CSR_matrix
      o row: corrseponds to word-frequency count

    - parameterArrayD: dictionary whose
      o key: topic
      o value: a numpy array of size |lexicon|
    - topicList: a list of topics
    - evalScoreF: function that takes in
      o word-frequency numpy array (for /a/ testing sentence)
      o topic parameter numpy array (for topic multinomial model)
    returns 
      o a number that assigns score to that topic
    Typically, this is *model.computeLogLikelihood function

    returns a list of topics with highest score
    - size = # rows in CSR_matrix
    '''
    predictedTopicL = []
    # number of documents
    numDocs = testingMat.get_shape()[ 0 ]
    for docId in range( numDocs ):
        scoreD = {} # key: topic, value: score evaluated for topic
        currentDoc = testingMat.getrow( docId )
        currentDoc = currentDoc.sum( axis = 0 )
        docWordFreqArray = fromOneDimMatrixToArray( currentDoc )
        for topic in topicList:
            scoreD[ topic ] = evalScoreF( docWordFreqArray, \
                                          parameterArrayD[ topic ] )
        maxTopic = max( scoreD.items(), key=operator.itemgetter(1) )[0]
        predictedTopicL.append( maxTopic )
    return predictedTopicL


# --------------------------
def plotFigure( confusionMat, topicIdxToStrFName, isTwentyNewsgroup ):
    '''
    Plots a heatmap confusion matrxi

    returns
    '''
    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )

    if isTwentyNewsgroup:
        topicIdxToTopicD = mapFromTopicIdxToTopic( topicIdxToStrFName )
        topicList = list( topicIdxToTopicD.keys() )
        topics = [ topicIdxToTopicD[i] for i in topicList ]
    else:
        topics = range( 104 ) # HARDCODED for industry_sector

    # heatmap
    heatmap = ax.matshow( confusionMat, cmap="Reds", aspect="equal" )
    ax.set_xticks( np.arange( len( confusionMat ) ) )
    ax.set_yticks( np.arange( len( confusionMat ) ) )
    ax.set_xticklabels( topics )
    ax.set_yticklabels( topics )
    # setting up rotation
    plt.setp( ax.get_xticklabels(), rotation=45, ha='left', va='top', \
            rotation_mode='anchor' )
    plt.ylabel( 'True topic' )
    plt.xlabel( 'Predicted topic' )
    plt.colorbar( heatmap )
    plt.show()

def reportPrecision( confusionMatL ):
    '''
    Returns precision mean and stdev from a list of ConfusionMatrices
    '''
    precisionL = []
    for idx, confusionMat in enumerate( confusionMatL ):
        tPCount = 0
        fPCount = 0
 
        for topicIdx, topicRow in enumerate( confusionMat ):
            tP = confusionMat[ topicIdx, topicIdx ]
            fP = sum( confusionMat[:, topicIdx] ) - tP
            tPCount += tP
            fPCount += fP
        totalCount = tPCount + fPCount
        precision = tPCount / float( totalCount )
        precisionL.append( precision )
    print( 'Precision: ', mean( precisionL ), ' +/- ', stdev( precisionL ) )
    return precisionL

def mapFromTopicIdxToTopic( topicFName ):
    '''
    topicFName is a file that contains
    a dictionary whose
    key: topic string
    val: ... DONT care

    returns a dictionary where 
    key: topicIDx
    val: topic string
    '''
    idxToStrD = {}
    with open( topicFName, 'rb' ) as f:
        topicKeyD = pickle.load( f )
    for topicIdx, (topic, rest) in enumerate( list( topicKeyD.items() ) ):
        idxToStrD[ topicIdx ] = topic
    return idxToStrD

# ------------------
# Per Split computation
def splitResults( splitNumber, smoothingParam=0.01, testCluster='twenty_newsgroups' ):
    '''
    Given a training/testing data split number, and smoothing param
    returns 3 things
    - a dictionary of trained ML parameter Array
    - prediction list for training set given the parameters
    - actual topic list for training set
    '''
    computeLogLikelihood = multinomialModel.computeLogLikelihood
    trainParameterGivenTopic = multinomialModel.trainParameterGivenTopic
    # --------------------------------
    # Step 1. Read and load training/test data split
    #splitNumber = 0
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster ) 
    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True, testCluster=testCluster ) 
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False, testCluster=testCluster ) 
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True, testCluster=testCluster ) 

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
    # key: topic, val: matrix whose col dimensions = |lexicon|, row dim = # docs belonging to topic
    
    trainingWordFrequencyD = {}
    # key: topic, value: array of topic wide frequency, len( array ) = | lexicon |
    for topic in topicList:
        # sum frequency row-wise
        topicFrequencyMat = partitionedTrainingD[ topic ].sum( axis=0 )
        trainingWordFrequencyD[ topic ] = fromOneDimMatrixToArray( topicFrequencyMat )

    # ------------------------------
    # Step 3. Derive ML estimates from training data (per topic)
    mlEstimatesD = {}
    for topic in topicList:
        mlEstimatesD[ topic ] = trainParameterGivenTopic( trainingWordFrequencyD[ topic ], SMOOTH )
        # BUG: note that ML estimates are NOT exactly one
        #print( np.sum( mlEstimatesD[topic] ) )

    # ------------------------------
    # Step 4. from test data, compute predictions of trained model 
    startTime = time.time()
    predicted = predict( testingMat, mlEstimatesD, topicList, computeLogLikelihood )
    endTime = time.time()
    print( "Elapsed Time for predicting : ", endTime-startTime )

    isIndustrySector = testCluster == 'industry_sector' 
    actual = [i-100 for i in testingLabel] if isIndustrySector else testingLabel

    return mlEstimatesD, predicted, actual


# -------------------------
# Main Function

if __name__ == '__main__':
    SMOOTH = 0.01
    '''
    Step 1. Read and load training/test data split
    Step 2. partition training/test data per topic
    Step 3. Derive ML estimates from training data (per topic)
    Step 4. from test data, compute predictions of trained model
    Step 5. Compute confusion matrix
    Step 6. Plot confusion matrix and report

    To simply plot existing confusion matrix, skip Step 4,5
    '''
    totalNumSplits = int( sys.argv[1] )
    TEST_CLUSTER = sys.argv[2]

    # Steps 1-4 condensed into splitResults function
    #totalNumSplits = 10
    predictedL = []
    actualL = []
    for i in range( totalNumSplits ):
        (mlEstimatesD, predicted, actual) = splitResults( i, SMOOTH, TEST_CLUSTER )
        predictedL.append( predicted )
        actualL.append( actual )
    # mlEstimatesD.. is not needed now, but will be needed
    # for updating using gradientDescent

    # -----------------------------
    # Step 5. Compute confusion matrix

    # For Confusion Matrix Writing
    confusionDataDir = 'confusionMatrix-multi'
    confusionMatrixFList = [] # list of strings
    confusionMatrixL = [] # list of confusion matrices for CV
    for i in range( totalNumSplits ):
        # files to write out to
        confusionMatrixF = getFilename( i, isTraining=False, isLabel=False, dataDirName=confusionDataDir, testCluster=TEST_CLUSTER )
        confusionMatrixFList.append( confusionMatrixF )

        # actual confusion matrices
        actual = actualL[ i ]
        predicted = predictedL[ i ]
        confusion_matrix_unnorm = metrics.confusion_matrix( actual, predicted )#, labels=topicList )
        confusionMatrix = confusion_matrix_unnorm.astype( 'float' ) /\
                        confusion_matrix_unnorm.sum( axis=1 )[:, np.newaxis]
        confusionMatrixL.append( confusionMatrix )
    
    # will fail if file exists 
    # creates file if it hasn't existed before
    for i in range( totalNumSplits ):
        confusionMatrix = confusionMatrixL[ i ]
        confusionMatrixF = confusionMatrixFList[ i ]
        try: 
            with open( confusionMatrixF, 'xb' ) as f:
                pickle.dump( confusionMatrix, f, protocol=pickle.HIGHEST_PROTOCOL )
        except IOError:
            with open( confusionMatrixF, 'wb' ) as f:
                pickle.dump( confusionMatrix, f, protocol=pickle.HIGHEST_PROTOCOL )
    # ----------------------------
    # Step 6. Plot confusion matrix and report
    # code from Samy

    avgConfusionMatrix = np.zeros( confusionMatrixL[0].shape )
    for mat in confusionMatrixL:
        for rowIdx, row in enumerate( mat ):
            for colIdx, entry in enumerate( row ):
                avgConfusionMatrix[ rowIdx, colIdx ] += float( entry / totalNumSplits )

    topicDFName = './data_pickles/twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'
    isTwentyNewsgroup = TEST_CLUSTER == 'twenty_newsgroups'
    plotFigure( avgConfusionMatrix, topicDFName, isTwentyNewsgroup )
    reportPrecision( confusionMatrixL )

