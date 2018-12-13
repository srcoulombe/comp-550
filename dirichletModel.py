import numpy as np
from numpy import log

# reporting
import time
def fromOneDimMatrixToArray( oneDimMatrix ):
    '''
    converts [1xn] matrix into a numpy array of shape( n, )

    returns a numpy array
    '''
    return np.array( oneDimMatrix )[ 0 ]

# ---------------------------
# Model Specific Functions
'''
 Functions called in *classification.py
 1. trainParameterGivenTopic( docWordFrequencyMat, smoothingParam, numDoscPerUpdate, maxIter, powerThreshold ): returns a 4-tuple
  - 1st tuple coord: is a numpy array of parameters, with shape[0] = numCols( docWordFrequencyMat
  - 2nd,3rd,4th coord: reported totalTime, numIter, numDocsUsed for training

 2. computeLogLikelihood( testWordFrequencyArray, parameterArray ): returns logLikelihood value of given document, based on parameterArray

 Helper Functions
 1. printArrayInfo( numpyArray, arrayName ): prints out max, min, and sum of array
 2. smoothArray( unsmoothedParamArray, smoothingParam ): returns a smoothed numpy array
 3. diPoch( x,y ): computes digamma( sum of args ) - digamma( x ), special care is taken when y[i] == 0
 4. getNumDenom( xMat, currentParameters ): returns numerator array, denominator scalar for DCM model parameter approximation
 5. getNumDenom_baseline( xMat, currentParameters ): same as 3, slower than 3.
 6. updateParameter( docWordFrequencyMat, numDocsPerUpdate, currentParameters, numDocs ): function called iteratively that returns a numpy array of new paramters
'''
# --------------------------
# Functions called by *classification.py
def trainParameterGivenTopic( docWordFrequencyMat, smoothingParam = 0, numDocsPerUpdate=1, maxIter=1000, powerThreshold = -6 ):
    '''
    Given a CSR matrix whose ith row corrseponds to an array of freq of
    in document, and a smoothing param and a numDocsPerUpdate, and a maxIter
    returns a 4-tuple 
    - a list of parameters that are ML estimates of prob occurence
    for each word
    - 3 scalars of reporting statistics

    one iteration of updates parameter with numDocsPerUpdate
    Assumes fixed-point algorithm (gradient ascent)

    returns a numpy array of parameters:
    - size = len( docWordFrequencyArray column)
    - values are ML estimates of dirichlet alpha parameters
    AND a 3-tuple:
    ( timeTakenToTrainThisTopic, numIterationsToTrain, numDocPerUpdate )
    '''
    thresholdVal = 10 ** (powerThreshold)
    matDim = docWordFrequencyMat.get_shape()
    lexiconSize = matDim[1]
    numDocs = matDim[0]
    #numDocsPerUpdate = numDocs# temporarily override
    #assert( numDocsPerUpdate <= numDocs )
    
    # initialize parameter and then update using fixed-point algorithm
    # essentially a thresholding method
    newAlpha = np.ones( lexiconSize ) * 2
    startTime = time.time()
    actualNumIter = maxIter
    for iterNum in range( maxIter ):
        # smoothing must happen per iteration
       
        oldAlpha = smoothArray( newAlpha, smoothingParam )
        updateDocIdxList = [( iterNum * numDocsPerUpdate + j ) % numDocs for j in range( numDocsPerUpdate) ]
        updateSubM = docWordFrequencyMat[ updateDocIdxList, : ] 
        newAlpha = updateParameter( updateSubM, numDocsPerUpdate, oldAlpha, numDocs )
            
        if( np.amax( np.absolute( newAlpha - oldAlpha ) ) < thresholdVal ):
            actualNumIter = iterNum
            break
        # otherwise, continue to update
    endTime = time.time()
    # SMOOTHING and sanity checks
    newAlpha = smoothArray( newAlpha, smoothingParam )
     
    return (newAlpha, endTime-startTime, actualNumIter, actualNumIter * numDocsPerUpdate)        
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
    logFirstFraction = - log( poch ( parameterArray.sum(), testWordFrequencyArray.sum() ) )
    logSecondFractionArray = log( poch ( parameterArray, testWordFrequencyArray ) )
    return logFirstFraction + logSecondFractionArray.sum()



# --------------------------
# Helper Functions
def printArrayInfo( numpyArray, arrayName ):
    '''
    Print max, min, sum of array, given the Array and its string name
    '''
    print( arrayName, " statistics" )
    print( "Max of array: ", np.amax( numpyArray ) )
    print( "Min of array: ", np.amin( numpyArray ) )
    print( "Sum of array: ", numpyArray.sum() )
    print( "num nonzeros: ", len( np.nonzero( numpyArray )[0] ) )
    return


def smoothArray( unsmoothedParamArray, smoothingParam ):
    '''
    given a numpy array of non-negative array, and a smoothingParam
    perform smoothing

    smootheArray[i] = unsmootheParamArray[i] + smoothingParam * (smallest positive unsmootheParam )
    returns a numpy array of size unsmootheParamArray.shape[0]
    '''
    nonZeroIdxArray = np.nonzero( unsmoothedParamArray )[0]
    smallestNonZeroVal = np.amin( unsmoothedParamArray[ nonZeroIdxArray ] )
    return unsmoothedParamArray + smoothingParam * smallestNonZeroVal

from scipy.special import digamma
def diPoch( x, y ):
    '''
    Given two Arrays
    - x: a numpy array with strictly positive values
    - y: a numpy array with non-negative values
    returns a numpy array that is roughly:
     - digamma( x+y ) - digamma( x )
     - special attention is paid for y(i) = 0

    '''
    outDiPochhammer = np.zeros( x.shape[0] )
    nonZeroIdx = np.nonzero( y )
    outDiPochhammer[ nonZeroIdx ] = digamma( x[ nonZeroIdx ] + y[ nonZeroIdx ] ) - \
                                    digamma( x[ nonZeroIdx ] )

    return outDiPochhammer

# getNumDenom is faster than getNumDenom_baseline by 1.6 - 2x times
def getNumDenom( xMat, currentParameters ):
    '''
    Same spec as other getNumDenom*
    Given a xMat (CSR matrix) whose dimensions are numDocsPerUpdate x len(currentParameters)
    and currentParameters
    - xMat is the document word frequency submatrix, corresponding to topic
    - currentParameters is a numpy array of dirichlet-multinomial parameters

    return 
    - a numpy array with | currentParameters | column
    - and denominator scalar
    '''
    numDocsPerUpdate, lexiconSize = xMat.get_shape()
    #numpy matrices
    alphaMat = np.tile( currentParameters , ( numDocsPerUpdate, 1 ) )
    digammaAlphaMat = np.tile( digamma( currentParameters ), ( numDocsPerUpdate, 1 ) )
    # result is a dense matrix
    nonZeroIdx = xMat.nonzero()
    firstTermMat = xMat + alphaMat

    # compute digamma only for non-zero values
    numMat = np.tile( np.zeros( lexiconSize ), (numDocsPerUpdate,1) )
    numMat[ nonZeroIdx ] = digamma( firstTermMat[ nonZeroIdx ] ) - digammaAlphaMat[ nonZeroIdx ]
    numeratorArray = np.array( numMat.sum( axis=0 ) )
    # denom calculation
    sumParam = currentParameters.sum()
    denomArray = fromOneDimMatrixToArray( xMat.sum( axis=1 ).transpose() ) + sumParam
    digammaSum = digamma( sumParam ) 
    denomScalar = digamma( denomArray ).sum() - numDocsPerUpdate * digammaSum
    return ( numeratorArray, denomScalar )

def getNumDenom_baseline( xMat, currentParameters ):
    '''
    Given a xMat (CSR matrix) whose dimensions are numDocsPerUpdate x len(currentParameters)
    and currentParameters
    - xMat is the document word frequency submatrix, corresponding to topic
    - currentParameters is a numpy array of dirichlet-multinomial parameters

    return 
    - a numpy array with | currentParameters | column
    - and denominator scalar
    '''
    numDocsPerUpdate, lexiconSize = xMat.get_shape()
    numeratorMat = np.zeros( (numDocsPerUpdate,lexiconSize) )
    denomArray = np.zeros( numDocsPerUpdate  )
    sumParameters = currentParameters.sum()
    for docIdx in range( numDocsPerUpdate ):
        docWordFrequencyArray = fromOneDimMatrixToArray( xMat.getrow( docIdx ).sum( axis=0 ) )
        allZeroDocument = np.nonzero( docWordFrequencyArray )[0].shape[0] == 0
        numeratorMat[ docIdx, : ] = diPoch( currentParameters, docWordFrequencyArray )
        denomArray[ docIdx ] = diPoch( np.array( [sumParameters] ), np.array( [docWordFrequencyArray.sum()] ) )
    # add by column
    numeratorArray = numeratorMat.sum( axis=0 )
    denomScalar = denomArray.sum() #denominator is scalar with respect to word
    return (numeratorArray, denomScalar)

def updateParameter( docWordFrequencyMat, numDocsPerUpdate, currentParameters, numDocs ):
    '''
    given a CSR matrix with numDocsPerUpdate number of rows, with |currentParameters| column
    and numDocsPErUpdate, and currentParameters numpy Array

    return new dirichlet parameters by fixed-point algorithm

    Assumes fixed-point algorithm

    returns a numpy array of parameters:
    - size = len( currentParameters )
    '''
    sumParameters = currentParameters.sum()
    lexiconSize = currentParameters.shape[0]
    newParameters = currentParameters

    numeratorArray, denomScalar = getNumDenom( docWordFrequencyMat, currentParameters )
    # BUG: when using 1 doc per update, and encountering a BUGGY file with empty word count,
    #      updated parameters are NaN
    # FIX: simply ignore update based on this faulty document
    if denomScalar == 0:
        # means that should return old parameter
        return currentParameters
    newParameters = currentParameters * numeratorArray / denomScalar # floatdivision
    return newParameters

