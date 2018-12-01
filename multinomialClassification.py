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

def plotFigure( confusionMat, topicList ):
    '''
    Plots a heatmap confusion matrxi

    returns
    '''
    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=1 )

    # heatmap
    heatmap = ax.matshow( confusionMat, cmap="Reds", aspect="equal" )
    ax.set_xticks( np.arange( len( confusionMat ) ) )
    ax.set_yticks( np.arange( len( confusionMat ) ) )
    ax.set_xticklabels( topicList )
    ax.set_yticklabels( topicList )
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

# ---------------------
# Deprecated Functions
# NOT USED
def reformFromListOfTuplesToDictionary( listOfTuples ):
    '''
    reform [(a1,b1), ... ] into d[a1] = b1

    returns a dictionary
    '''
    return dict( listOfTuples )

def d_computeLogLikelihood( testSentence, parameters, wordList ):
    '''
    Given testSentence, and parameters for a multinomial distribution
    calculate log-likelihood of seeing test sentence
    assumes Naive Bayes assumption
    
    - testSentence: word-frequency count dictionary
    - parameters: a list of size |lexicon|, corresponding to values of multinomial parameter

    returns a number
    '''
    wordFrequencyList = []
    for word in wordList:
        countInTestSentence = 0
        if word in testSentence.keys():
            countInTestSentence = testSentence[ word ]
        # ow zero count
        wordFrequencyList.append( countInTestSentence )
    logParam = np.log( parameters )
    return np.dot( np.array( wordFrequencyList ), logParam )


# NOT USED, but INSPIRED FROM
def d_trainParameterGivenTopic( inputCorpus, wordList, smoothingParam = 0 ):
    '''
    Given a dictionary of word-frequency count for a topic corpus, and wordList
    return a list of ML parameters

    o inputCorpus: training dictionary whose
     - key: string word
     - value: frequency count of word in topic corpus
    o wordList: list of words

    returns a list of parameters:
     - size = |wordList|
     - values are ML estimates of prob occurence for each word
    '''
    totalNumWordsInCorpus =  float( sum( list( inputCorpus.values() ) ) )
    lexiconSize = len( wordList )
    denominator = totalNumWordsInCorpus + lexiconSize * smoothingParam
    mlEstimateList = []
    topicCount = 0
    for word in wordList:
        if( word in inputCorpus.keys() ): topicCount = inputCorpus[ word ]
        else: topicCount = 0
        numerator = smoothingParam + topicCount  
        mlEstimateList.append( numerator / denominator ) 
    return mlEstimateList
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
    splitNumber = int( sys.argv[1] )
    isDump = sys.argv[2] == 'dump'
    # --------------------------------
    # Step 1. Read and load training/test data split
    #splitNumber = 0
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False ) 
    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True ) 
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False ) 
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True ) 
    # For Confusion Matrix Writing
    confusionDataDir = 'confusionMatrix'
    confusionMatrixF = getFilename( splitNumber, isTraining=False, isLabel=False, dataDirName=confusionDataDir, alternateClusterName='' )
     
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
    # actually don't think I need this
    (_, partitionedTestingD) = partitionDataPerTopic( testingLabel, testingMat )
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


    # -----------------------------
    # Step 5. Compute confusion matrix
    confusion_matrix_unnorm = metrics.confusion_matrix( actual, predicted, labels=topicList )
    confusionMatrix = confusion_matrix_unnorm.astype( 'float' ) /\
                        confusion_matrix_unnorm.sum( axis=1 )[:, np.newaxis]

    # will fail if file exists 
    # creates file if it hasn't existed before
    if isDump:
        try: 
            with open( confusionMatrixF, 'xb' ) as f:
                pickle.dump( confusionMatrix, f, protocol=pickle.HIGHEST_PROTOCOL )
        except IOError:
            with open( confusionMatrixF, 'wb' ) as f:
                pickle.dump( confusionMatrix, f, protocol=pickle.HIGHEST_PROTOCOL )
    # ----------------------------
    # Step 6. Plot confusion matrix and report
    with open( confusionMatrixF, 'rb' ) as f:
        confusionMatrix = pickle.load( f )
    
    # code from Samy
    #plotFigure( confusionMatrix, topicList )
    confusionMatrixFL = []
    confusionMatrixL = []
    for i in range( 3 ):
        confusionMatrixFL.append( getFilename( i, isTraining=False, isLabel=False, dataDirName=confusionDataDir, alternateClusterName='' ) )
    for confusionMatrixFile in confusionMatrixFL:
        with open( confusionMatrixFile, 'rb' ) as f:
            confusionMatrixL.append( pickle.load( f ) )
    reportPrecision( confusionMatrixL )
    '''
    numSamples = 10
    for i in range( numSamples ):
        actualTopic = 3
        likelihoodD = {}
        currentSentence = partitionedTestingD[ actualTopic ].getrow( i )
        currentSentence = currentSentence.sum( axis=0 )
        sentWordFreqArray = fromOneDimMatrixToArray( currentSentence) 
        for topic in topicList:
            likelihoodD[ topic ] =  computeLogLikelihood( sentWordFreqArray, mlEstimatesD[ topic ] )
        maximizingTopic = max( likelihoodD.items(), key=operator.itemgetter( 1 ) ) [ 0 ]
    '''
'''
if __name__ == '__main__':
    SMOOTH = 0.01
    corpusWordFrequencyName = '20_newsgroup_tokens_and_frequencies.pickle'
    topicWordFrequencyName = '20_newsgroup_tokens_and_frequencies_by_topics_dictionary.pickle'
    docWordFrequencyName = 'twenty_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'

    topicToDocIdName = 'twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'

    with open( topicToDocIdName, 'rb' ) as f:
        topicToDocId = pickle.load( f )
    with open( corpusWordFrequencyName, 'rb' ) as f:
        corpusWordFrequency = pickle.load( f )
    with open( topicWordFrequencyName, 'rb' ) as f:
        topicWordFrequency = pickle.load( f )
    with open( docWordFrequencyName, 'rb' ) as f:
        docWordFrequency = pickle.load( f )
    firstDoc = list( docWordFrequency.keys() )[0]
    print( firstDoc ) # this is a string like topic+fileID
    lexicon = list( corpusWordFrequency.keys() )

    contentOfFirstDoc = topicToDocId[ 'alt.atheism' ][firstDoc]
    print( contentOfFirstDoc )
    # Train ML parameters from corpus
    topicParams = []
    topicList = list( topicWordFrequency.keys() )

    # TODO: split training-test-validation test data.. do it by # document proportions per topic
    for topic in topicList: 
        topicParams.append( trainParameterGivenTopic( dict( topicWordFrequency[ topic ] ), lexicon, SMOOTH ) )

    # compute likelihood for each topic
    likelihoodD = {}
    for topicIdx in range( len( topicList ) ):
        topic = topicList[ topicIdx ]
        likelihoodD[ topic ] = computeLogLikelihood( docWordFrequency[ firstDoc ], topicParams[ topicIdx ], lexicon )
    maximizingTopic = max( likelihoodD.items(), key=operator.itemgetter( 1 ) ) [ 0 ] 
    print( "Trained model predicts that for sentence above  topic ", maximizingTopic, ' is the topic for this doc' )
    print( likelihoodD )
'''


