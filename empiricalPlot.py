import pickle
from collections import Counter
import matplotlib.pyplot as plt
from scipy import interpolate
#unit testing would be great
def interpolate( countD ):
    '''
    Given a dict
    key: docID
    value: count
    make points: (count, #docs with count) and interpolate them
    
    return interpolation function
    '''
    print( countD )
    countTupleList = Counter( countD.values() ).most_common()
    print( countTupleList ) 
    return 
def computeCountPerWord( wordCountD, wordList, totalNumDoc ):
    '''
    given totalNumDoc, wordCountD which is {'word': {'docID': count}}
    returns a function that takes in string word, and a word count
    returning a number between [0,1]
    '''
    '''
    DOC_MAX = 5
    MIN = 0
    COUNT_MAX = 30
    '''
    for word in wordList:
        print( word )
        docCounts = wordCountD[ word ]
        #print( docCounts )
        # key: docID, value: frequency of word in docID
        #interpolate( docCounts )
        # interpolation
        numBins = 150
        scalingFactor = 1
        plt.hist( list( docCounts.values() ) , numBins )
        #logscale
        plt.yscale( 'log' )
        minY = 0
        plt.axis( [0, numBins, minY, len( list( docCounts.keys() ) )/scalingFactor ] )
        plt.show()
    return 
def computeAvgCountPerClass( wordList, g ):
    '''
    given a wordList
    g is a function: takes in string word, and a word count
    , returning a number between [0,1]

    returns a function over word counts, where
    f( word count ) = sum( g( word count, word ) ) / number of words in wordList
    '''

def reformToDictionaryPerWord( perDocD, wordList ):
    '''
    Transforms { key_o: { key_i: val } } -> { key_i: { key_o: val } }

    Given a dictionary of dictionary, and a list of all words in dictionary:
     outer key: docID
        inner key: word
        inner value: word count in docID
    Return a dictionary of dictionary
     outer key: word
        inner key: docID
        inner value: word count in docID
    '''
    perWordD = {}
    for word in wordList:
        perWordD[ word ] = {}
    for docID, wordCount in perDocD.items() :
        for word, wordCountInDocID in wordCount.items():
           perWordD[ word ][ docID ] = wordCountInDocID

    return perWordD

def mostFrequentWords( listOfWordCount ):
    '''
    Given a list of tuples: 1st coord of tuple = word, 2nd coord = count in corpus
    returns a list of lists (of tuples),
    where 0th entry of list = list of 500 most frequent wordCount tuple (sz = 500)
    where 1st entry of list = list of next 5000 most frequent wordCount tuple (sz = 5000)
    where last entry = list of rest wordCount tuples
    '''
    listOfWordCount.sort( key = lambda wordCountTuple: wordCountTuple[ 1 ], reverse=True )
    return [ listOfWordCount[:500], listOfWordCount[500:5500], listOfWordCount[5500:] ]
    
if __name__ == '__main__':
    corpusFile = '20_newsgroup_tokens_and_frequencies.pickle'
    with open( corpusFile, 'rb' ) as f:
        wordCountTupleD = pickle.load( f )
        wordCountTupleList = list( wordCountTupleD.items() )
    classOne, classTwo, classThree = mostFrequentWords( wordCountTupleList )
    # classWords is a list of lists of words
    # 0th entry = most frequent
    classWords = []
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classOne ] )
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classTwo] )
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classThree] )
    
    # ----------------
    # unit tests for mostFrequentWords, reformToDictionaryPerWord
    
    #print( classWords[0] )
    '''d = {0: {'the': 1, 'hello': 2}, 2: {'yellow': 3, 'the': 1}, 4: {'the': 3}}
    wordList = ['the', 'hello', 'yellow']
    numDocs = 3
    '''
    wordList = classWords[0] + classWords[1] + classWords[2]
    perDocDFilename = 'dict_of_dicts_docID_token_freq_dicts.pickle'
    with open( perDocDFilename, 'rb' ) as f:
        d = pickle.load( f )
    firstKey = list( d.keys())[0]
    numDocs = len( list( d.keys() ) )
    reformedD =  reformToDictionaryPerWord( d, wordList )
    computeCountPerWord( reformedD, wordList[:5], numDocs)
