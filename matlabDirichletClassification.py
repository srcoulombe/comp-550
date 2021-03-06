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

import scipy.io
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
import dirichletModel

# ----------------------------
# Generic function
def predict( testingMat, parameterArrayD, topicList, evalScoreF ):
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
    Plots a heatmap confusion matrix, given isTwentyNewsgroup
    topicIdxToStrFName: if isTwentyNewsgroup, this file is used to map 
     -topicIdx -> topic string name
    otherwise, this file is ignored

    returns a plot
    '''
    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )

    # returned result from test_cluster
    # is a list of topics (either a list of string OR a list of numbers)
    if isTwentyNewsgroup:
        topicIdxToTopicD = mapFromTopicIdxToTopic( topicIdxToStrFName )
        #topicList = list( topicIdxToTopicD.keys() )
        topicList = [i for i in topicIdxToTopicD.keys() if int(i) != 16 ]
        print( topicList )
        topics = [ topicIdxToTopicD[i] for i in topicList ]
    else:
        topicList = range( 104 ) # HARDCODED, for industry sector
        topics = topicList

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
    if (len( precisionL ) == 1 ):
        print( 'Precision for one iteration: ', precisionL[0] )
    else:
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

def reportTrainAndTestStatistics( trainTestStatisticsL, maxIter ):
    '''
    Given a list of lists, where reports statistics for a particular split
    - 0th ele: is a list of time taken to train a topic
    - 1st ele: is a list of numIter taken to train each topic
    - 2nd ele: is a list of numDocs used to train per topic
    - 3rd ele: is a list of length 1, reporting time taken to generate test prediction

    AND a scalar maxIter

    Side Effect: prints out statistics per topic, for a splitNumber
    '''
    topicStatsTimeL = trainTestStatisticsL[ 0 ]
    topicStatsNumIterL = trainTestStatisticsL[ 1 ]
    topicStatsNumDocsL = trainTestStatisticsL[ 2 ]
    predictionTimeL = trainTestStatisticsL[ 3 ]

    avgTime = mean( topicStatsTimeL )
    stdevTime = stdev( topicStatsTimeL )
    avgIter = mean( topicStatsNumIterL )
    stdevIter = stdev( topicStatsNumIterL )
    avgDocs = mean( topicStatsNumDocsL )
    stdevDocs = stdev( topicStatsNumDocsL )
    timeToPredictTest = predictionTimeL[0]
    print()
    print( "Avg time to train model per topic: ", avgTime, "+- ", stdevTime )
    print( "Avg time per iteration is: ", avgTime / float( avgIter ) )
    print( "Avg time per document is: ", avgTime / float( avgDocs ) )
    print( "Total time taken to update topics is", sum( topicStatsTimeL ) )
    print( "Avg number of iterations per topic: ", avgIter , "+- ", stdevIter )
    print( "Max NumIter was: ", maxIter )
    print( "Avg Num documents used for training per topic: ", avgDocs, "+- ", stdevDocs )
    print( "Elapsed Time for predicting dirichlet: ", timeToPredictTest )
    return

# ------------------
# Per Split computation
def splitResults( splitNumber, smoothingParam=0.01, maxIter=1000, numDocsPerUpdate=1, powerThreshold=-6, testCluster='twenty_newsgroups' ):
    '''
    Given a training/testing data split number, and smoothing param
    maxIter, numDocsPerUpdate, powerThreshold, and testCluster name

    returns 3 things
    - a dictionary of trained ML parameter Array
    - prediction list for training set given the parameters
    - actual topic list for training set

    '''
    trainParameterGivenTopic = dirichletModel.trainParameterGivenTopic
    computeLogLikelihood = dirichletModel.computeLogLikelihood
    smoothArray = dirichletModel.smoothArray
    # --------------------------------
    # Step 1. Read and load training/test data split
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster )
    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True, testCluster=testCluster )
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False ,testCluster= testCluster )
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True ,testCluster= testCluster )

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
   
    # for Dirichlet, we don't need topic wide frequency
    trainingWordFrequencyD = partitionedTrainingD
    # ------------------------------
    # Step 3. Derive ML estimates from training data (per topic)
    # MATLAB .mat file to write out pickled matrix
    trainingMatlabF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster, extensionName='mat' )
    newPartitionedTrainingD = {'arr' + str(k):v for k,v in partitionedTrainingD.items() }
    scipy.io.savemat( trainingMatF, mdict=newPartitionedTrainingD )
    # trained parameter filename
    '''
    trainingParamF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster, extensionName='mat', isUsingMatlab=True )
    mlEstimatesMat = scipy.io.loadmat( trainingParamF )['alphaLearntM'] 
    mlEstimatesMat = np.matrix( mlEstimatesMat ) 
    mlEstimatesD = {}
    '''
    trainingParamF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster, extensionName='pickle', isUsingMatlab=True )
    with open( trainingParamF, 'rb' ) as f:
        mlEstimatesD = pickle.load( f )
    topicStatsTimeL = []
    topicStatsNumIterL = []
    topicStatsNumDocsL = []
    
    # modify topicList and testingMat to exclude m
    topicList = [ i for i in topicList if int(i) != 16 ]

    for topic in topicList:
        # trainingWordFrequencyD[ topic ]
        # is a numpy array for multinomial
        # is a submatrix for dirichlet
        #mlEstimatesD[topic] = smoothArray( np.array( mlEstimatesMat[ topic ] ), smoothingParam )
        mlEstimatesD[ topic ] = smoothArray( mlEstimatesD[ topic ], smoothingParam )
        timeTakenToTrain = 120 # on average seems to be 120 sec per topic
        numIterToTrain = 1000
        numDocumentsToTrain = numIterToTrain * 800
        
        topicStatsTimeL.append( timeTakenToTrain )
        topicStatsNumIterL.append(numIterToTrain)
        topicStatsNumDocsL.append( numDocumentsToTrain)
        # BUG: note that ML estimates are NOT exactly one
        #print( np.sum( mlEstimatesD[topic] ) )
 
    trainTestStatisticsL = [] # a list of lists
    trainTestStatisticsL.append( topicStatsTimeL )
    trainTestStatisticsL.append( topicStatsNumIterL)
    trainTestStatisticsL.append( topicStatsNumDocsL )
    # ------------------------------
    # Step 4. from test data, compute predictions of trained model
    startTime = time.time()
    # skip prediction for now

    # from testingLabel, get firstIdx
    idxToRemove = [ idx for idx,ele in enumerate( testingLabel ) if int(ele) == 16 ]
    mask = np.ones( testingMat.shape[0], dtype=bool )
    mask[ idxToRemove ] = False
    testingMat = testingMat[ mask ]

    predicted = predict( testingMat, mlEstimatesD, topicList, computeLogLikelihood )
    #predicted = [1 for i in range(  testingMat.get_shape()[0] ) ]
    endTime = time.time()
    trainTestStatisticsL.append( [endTime-startTime] )

    isIndustrySector = testCluster == 'industry_sector'
    # matlab version was not reporting topic 17 (idx16) correctly
    testingLabel = [ i for i in testingLabel if i != 16 ]
    actual = ( [ i - 100  for i in testingLabel ] ) if isIndustrySector else testingLabel
    #print( actual )
    return mlEstimatesD, predicted, actual, trainTestStatisticsL


# -------------------------
# Main Function

if __name__ == '__main__':
    NUM_VOCAB = 255669
    #SMOOTH = 0.01 / NUM_VOCAB # reason for doing this the total size of alpha should increase by 1 percent after smoothing
    SMOOTH = 0.85
    MAX_ITER = 1
    NUM_DOCS_PER_UPDATE = 1

    totalNumSplits = int(sys.argv[1])
    POWER_THRESHOLD = -6
    SMOOTH = float(sys.argv[2])
    TEST_CLUSTER = sys.argv[3]
    isTwentyNewsgroup = TEST_CLUSTER == 'twenty_newsgroups'
    # threshold for stopping FP will be if max( abs( alpha_diff ) ) <= 10**(POWER_THRESHOLD)
    '''
    Step 1. Read and load training/test data split
    Step 2. partition training/test data per topic
    Step 3. Derive ML estimates from training data (per topic)
    Step 4. from test data, compute predictions of trained model
    Step 5. Compute confusion matrix
    Step 6. Plot confusion matrix and report

    To simply plot existing confusion matrix, skip Step 4,5
    '''
    # Steps 1-4 condensed into splitResults function
    #totalNumSplits = 2
    predictedL = []
    actualL = []
    for i in range( totalNumSplits ):
        (mlEstimatesD, predicted, actual, trainTestStatisticsL) = splitResults( i, SMOOTH, MAX_ITER, NUM_DOCS_PER_UPDATE, POWER_THRESHOLD, TEST_CLUSTER )
        predictedL.append( predicted )
        actualL.append( actual )
        reportTrainAndTestStatistics( trainTestStatisticsL, MAX_ITER )
    # -----------------------------
    # Step 5. Compute confusion matrix

    # For Confusion Matrix Writing
    confusionDataDir = 'confusionMatrix' 
    confusionMatrixFList = [] # list of strings
    confusionMatrixL = [] # list of confusion matrices for CV
    for i in range( totalNumSplits ):
        # files to write out to
        confusionMatrixF = getFilename( i, isTraining=False, isLabel=False, dataDirName=confusionDataDir, testCluster=TEST_CLUSTER )
        confusionMatrixFList.append( confusionMatrixF )

        # actual confusion matrices
        actual = actualL[ i ]
        predicted = predictedL[ i ]
        assert( len( set( actual ) ) == 19 )
        confusion_matrix_unnorm = metrics.confusion_matrix( actual, predicted )
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
    plotFigure( avgConfusionMatrix, topicDFName, isTwentyNewsgroup )
    reportPrecision( confusionMatrixL )
