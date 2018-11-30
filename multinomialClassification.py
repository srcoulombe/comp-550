import pickle
import numpy as np
from numpy import log
import operator

# NOT USED
def reformFromListOfTuplesToDictionary( listOfTuples ):
    '''
    reform [(a1,b1), ... ] into d[a1] = b1

    returns a dictionary
    '''
    return dict( listOfTuples )

def computeLogLikelihood( testSentence, parameters, wordList ):
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

def trainParameterGivenTopic( inputCorpus, wordList, smoothingParam = 0 ):
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

def reformMatrixToWordFrequencyD( sparseMat ):
    '''
    Given a Dok Matrix
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

