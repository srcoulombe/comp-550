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
def computeLogLikelihood_multi( testWordFrequencyArray, parameterArray ):
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


from scipy.special import poch
# pochhammer function: input (x,y) -> returns a number defined as gamma(x+y) / gammy(x)
# or equivalently: gamma( sum of args ) / gamma( first arg )
def computeLogLikelihood( testWordFrequencyArray, parameterArray ):
    '''
    given a testWordFrequencyArray, and a parameterArray
    returns loglikelihood function

    - each Array is a numpy array with same length (length = size of lexicon)
    - both Arrays are aligned (i.e. tWFArray[i] corresponds to frequency of word i,
                                    pArray[i] corresponds to dirichlet parameter for word i )
    assumes dirichlet-multinomial
    '''
    logPochMinka = lambda x,y: log( poch( x, y ) )
    # input: 2-tuple x -> output: log(poch( x[0], x[1] ) )
    logPochMinkaTuple = lambda x: log( poch( x[0],x[1] ) )
    # computes log( gamma( sum of args ) / gamma( first arg  ) )
    logFirstFraction = - logPochMinka( parameterArray.sum(), testWordFrequencyArray.sum() )
    tupleIterableFromTwoArrays = zip( parameterArray, testWordFrequencyArray )
    logSecondFractionIterable = map( logPochMinkaTuple, tupleFromTwoArrays )  
    return logFirstFraction + sum( logSecondFractionIterable )

# ----------------------------
# Generic function
def predict( testingMat, parameterArrayD, topicList, evalScoreF=computeLogLikelihood ):
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
     in our case, this is computeLogLikelihood function

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
def plotFigure( confusionMat, topicIdxToStrFName ):
    '''
    Plots a heatmap confusion matrxi

    returns
    '''
    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )

    topicIdxToTopicD = mapFromTopicIdxToTopic( topicIdxToStrFName )
    topicList = list( topicIdxToTopicD.keys() )
    topics = [ topicIdxToTopicD[i] for i in topicList ]
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
def splitResults( splitNumber, smoothingParam=0.01 ):
    '''
    Given a training/testing data split number, and smoothing param
    returns 3 things
    - a dictionary of trained ML parameter Array
    - prediction list for training set given the parameters
    - actual topic list for training set
    '''
    # --------------------------------
    # Step 1. Read and load training/test data split
    #splitNumber = 0
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
    predicted = predict( testingMat, mlEstimatesD, topicList )
    endTime = time.time()
    print( "Elapsed Time for predicting : ", endTime-startTime )
    actual = testingLabel

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
    #splitNumber = int( sys.argv[1] )
    #isDump = sys.argv[2] == 'dump'

    # Steps 1-4 condensed into splitResults function
    totalNumSplits = 10
    predictedL = []
    actualL = []
    for i in range( totalNumSplits ):
        (mlEstimatesD, predicted, actual) = splitResults( i, SMOOTH )
        predictedL.append( predicted )
        actualL.append( actual )
    # mlEstimatesD.. is not needed now, but will be needed
    # for updating using gradientDescent

    # -----------------------------
    # Step 5. Compute confusion matrix

    # For Confusion Matrix Writing
    confusionDataDir = 'confusionMatrix'
    confusionMatrixFList = [] # list of strings
    confusionMatrixL = [] # list of confusion matrices for CV
    for i in range( totalNumSplits ):
        # files to write out to
        confusionMatrixF = getFilename( i, isTraining=False, isLabel=False, dataDirName=confusionDataDir, alternateClusterName='' )
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
    plotFigure( avgConfusionMatrix, topicDFName )
    reportPrecision( confusionMatrixL )

