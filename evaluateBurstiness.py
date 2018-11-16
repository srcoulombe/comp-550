import matplotlib

import random
import nltk
from scipy.stats import multinomial, binom
from nltk.corpus import words
nltk.download( 'words' )

# percentile calculation
import numpy as np

def bagOfWords( countVectorDict ):
    '''
    Given a dictionary whose 
    o keys: words
    o values: count of word

    return a mutinomial distribution d, where
    d(w) = count(w) / Sum( count(w) ) for all w
    '''
    listOfCount = list( countVectorDict.values() )
    sumCount = sum( listOfCount )
    listOfCount = [count / float( sumCount )  for count in listOfCount ] 
    return multinomial( sumCount, listOfCount )

def generateData( maxWordCount, minWordCount = 10**5, numCountsUpper=3 ):
    '''
    returns a dictionary
    o keys: words
    o values: count of word
    '''
    numWords = max( minWordCount, random.randrange( maxWordCount ) ) 
    # intuition: words shouldn't occur too frequently
    countUpper = max(3, numCountsUpper ) 
    # minimum 3 (avoids divide by zero)

    print( countUpper )
    wordList = random.sample( range( len( words.words() ) ), numWords )

    #wordList = random.sample( words.words(), numWords )
    countList = [ random.randrange( countUpper ) for i in range( numWords ) ]
    numCounts = sum( countList )

    print( "Generated numWords: ", numWords, " length of Document: ", sum( countList ) )
    return dict( zip( wordList, countList ) )

def commonAverageRareSplit( countVector, commonPercentile, averagePercentile):
    '''
    given dictionary key: word, values: counts of words
    and commonPercentile (e.g. 70%), averagePercentil(e.g. 20%)

    Split countVector into three dictionaries representing
    1st group: total count is >=70 % of counts
    2nd group: total count is next 20% of counts
    3rd group: rest

    return (dict, dict, dict) representing 1st, 2nd, 3rd group
    '''
    interpolation = 'nearest'
    wordCounts = list( countVector.values() )
    commonLb = np.percentile( wordCounts, commonPercentile, interpolation )
    commonCounts = [ count for count in wordCounts if count >= commonLb ]

    avgAndRareCounts = [ count for count in wordCounts if count < commonLb ]
    avgLb = np.percentile( avgAndRareCounts, averagePercentile, interpolation )
    
    rareCounts = [ count for count in avgAndRareCounts if count < avgLb ]
    avgCounts = [ count for count in avgAndRareCounts if count >= avgLb ]
    
def binomialProbability( numTrials, x_h, p_h ):
    '''
    Given numTrials experiment, calculate binomial probability
    that heads appeared x_h, p_h being heads probability

    returns a number between 0 and 1
    '''
    return binom.pmf( x_h, numTrials, p_h ) 

if __name__ == '__main__':
    minNumWordsInDocument = 10 ** 2
    perWordCountUpperLimit = 7
    WORDS = words.words()
    maxNumWordsInDocument = 10 ** 3
    # dictionary whose key: word, value: count of word
    artificialData = generateData( maxNumWordsInDocument, minNumWordsInDocument, perWordCountUpperLimit )
    multiDistribution = bagOfWords( artificialData )
    # number of dice throwing: sum of word counts

    # -------------------
    # Synthetic document generation:
    # randomly drawn a sample from multinomial distribution
    # means generating a document! (based on word vector count)
    samples =  multiDistribution.rvs( size = 1 )

    # ------------------
    # Printing document count vector
    # actual document may contain more than upperLimit seen in original vector count 
    wordIdxList = list(artificialData.keys())
    numDisplay = 10
    for i in range( numDisplay ):
        wordIdx = wordIdxList[ i ]
        print( "Word: ", WORDS[wordIdx], " count: ", samples[0][i] )
    print( sum(samples[0] ) )
