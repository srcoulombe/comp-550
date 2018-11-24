import pickle
from collections import Counter
import matplotlib.pyplot as plt
from scipy import interpolate

import numpy as np

'''
Helper Functions
 I. building empirical histograms:
   - interpolate (TODO): needs work and redefinition probably
   - computeCountPerWord( wordCountD, wordList, totalNumDoc, numBins )
   - computeAvgCountPerClass( wordList, g )
     o g is whatever is outputted from computCountPerWord
   - graphEmpiricalCounts( docCounts, displayString='none' )

II. Classifying word in corpus to 3 classes:
   - reformToDictionaryPerWord( perDocD, wordList )
   - mostFrequentWords( listOfWordCount )
      o listOfWordCount is a list of tuples: (word, word count in corpus)
'''
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

def computeCountPerWord( wordCountD, wordList, totalNumDoc, numBins ):
    '''
    given totalNumDoc
    o wordCountD which is {'word': {'docID': count}}
    o wordList: list of words in corpus
    o totalNumDoc: total number of documents in corpus
    o numBins: total number of word_count allowed, used to bin docCounts 
               from [0,1), ... [numBins-2, numBins-1]

    returns a dictionary that takes in key: string word
            returning a list of numbers where

            f[ word ][ word_count ] returns number of document count
            that had word_count # of word

            Dimension: roughly: # words * numBins
            word_count is VALID only from 0.. to numBins - 2
    '''
    wordHist = {}
    bins = range( numBins )
    for word in wordList:
        docCounts = wordCountD[ word ] 
        # key: docID, value: frequency of word in docID

        # ---------------
        # documents without any count of word, is not in docCounts
        # compensating for that
        nonzeroWordCountList = list( docCounts.values() )
        numZeroCountDocs = totalNumDoc - len( nonzeroWordCountList )
        zeroInclusiveCountList = nonzeroWordCountList + [ 0 for i in range( numZeroCountDocs ) ] 
        
        # TODO: work on interpolation
        #interpolate( docCounts )
        
        # alpha = 0 to remove drawing 
        #( docCounts, _, _ ) = plt.hist( zeroInclusiveCountList, bins, alpha = 0 )
        ( docCounts, _ ) = np.histogram( zeroInclusiveCountList, bins ) 
        wordHist[ word ] = docCounts.tolist() 
    return wordHist
    

def computeAvgCountPerClass( wordList, g  ):
    '''
    given a wordList
    g is a dictionary: takes in string word,
    value: list of word count, returning number repr # of docs that had i counts for word

    returns a list
    f[ i ] = sum_wordinList( g[word][ i ] ) / number of words in wordList
    '''
    # all values have same length
    numBins = len( list( g.values() )[ 0 ] )
    numWords = len( wordList )
    currentAvg = np.zeros( numBins )
    for word in wordList:
        currentAvg = np.add( currentAvg, np.array( g[ word ] ) ) 
    return ( currentAvg / numWords ).tolist()

def graphEmpiricalCounts( docCounts, displayString='none' ):
    '''
    graphs empirical count as found in docCounts

    string word
    wordHist is a list of counts associated with word, where ith count = # of docs which had word i times
    '''
    numBins = len( docCounts )
    xAxis = range( numBins )
    totalNumDoc = sum( docCounts )
    scalingFactor = 1
    #logscale
    plt.yscale( 'log' )
    minY = 0
    plt.axis( [0, numBins, minY,  totalNumDoc/scalingFactor ] )
    plt.scatter( xAxis, docCounts, label=displayString )
    plt.legend( loc='upper right' )
    plt.plot( xAxis, docCounts )
    #plt.show()

    return 


# --------------------------------------------------------------------------------
# Part II: classifying words into common, avg, and rare

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
   
#--------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    corpusFile = '20_newsgroup_tokens_and_frequencies.pickle'
    with open( corpusFile, 'rb' ) as f:
        wordCountTupleD = pickle.load( f )
        wordCountTupleList = list( wordCountTupleD.items() )
    # ---------------
    # classifying words into three categories
    classOne, classTwo, classThree = mostFrequentWords( wordCountTupleList )
    # classWords is a list of lists of words
    # 0th entry = most frequent
    classWords = []
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classOne ] )
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classTwo] )
    classWords.append( [ wordTuple[ 0 ] for wordTuple in classThree] )
    
    # ----------------
    # empirical Frequency count
    wordList = classWords[0] + classWords[1] + classWords[2]
    perDocDFilename = 'dict_of_dicts_docID_token_freq_dicts.pickle'
    with open( perDocDFilename, 'rb' ) as f:
        d = pickle.load( f )
    firstKey = list( d.keys())[0]
    numDocs = len( list( d.keys() ) )
    reformedD =  reformToDictionaryPerWord( d, wordList )

    # compute small amounts
    sliceList = []
    sliceLen = len( classOne )
    idx = 0
    sliceList.append( slice( idx, idx+sliceLen ))
    idx= sliceLen + idx
    sliceLen = len( classTwo )
    sliceList.append( slice( idx, idx+sliceLen ))
    idx= sliceLen + idx
    sliceLen = len( classThree )
    sliceList.append( slice( idx, idx+sliceLen ))
    
    classAvgList = [] # is a list of lists.. where classAvgList[0][i]: common 500 word avg for count of docs with word appearing i times
    '''
    for s in sliceList:
        wordHist = computeCountPerWord( reformedD, wordList[s], numDocs, numBins=50)
        smallClassAvg = computeAvgCountPerClass( wordList[ s ], wordHist )  
        classAvgList.append( smallClassAvg )
        print( 'finished for ', str( s ) )
    
    with open ( '20_newsgroup_empirical_count.pickle', 'wb' ) as handle:
        pickle.dump( classAvgList, handle, protocol=pickle.HIGHEST_PROTOCOL )
    '''
    
    with open ( '20_newsgroup_empirical_count.pickle', 'rb' ) as f:
        classAvgList = pickle.load( f )
    for classAvgIdx in range( len( classAvgList ) ):
        graphEmpiricalCounts( classAvgList[ classAvgIdx ], 'class rank ' + str( classAvgIdx ) ) 
    plt.show()
    #graphEmpiricalCounts( 'guess', wordHist[ 'guess' ] )
