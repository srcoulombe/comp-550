from splitData import getFilename, partitionDataPerTopic
import scipy.io, pickle
import sys
from scipy.sparse import csr_matrix
import numpy as np

if __name__ == '__main__':
    testCluster = sys.argv[1]
    splitNumber = int( sys.argv[2] )
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

    trainingLabelF = getFilename( splitNumber, isTraining=True, isLabel=True, testCluster=testCluster, extensionName='mat' )
    testingMatF = getFilename( splitNumber, isTraining=False, isLabel=False ,testCluster= testCluster, extensionName='mat' )
    trainingMatF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster, extensionName='mat' )
    testingLabelF = getFilename( splitNumber, isTraining=False, isLabel=True ,testCluster= testCluster, extensionName='mat' )
    
    
    (topicList, partitionedTrainingD) = partitionDataPerTopic( trainingLabel, trainingMat )
    newPartitionedTrainingD = {'arr' + str(k):v for k,v in partitionedTrainingD.items() }

    scipy.io.savemat( trainingMatF, mdict=newPartitionedTrainingD )
    # after some time

    trainingParamF = getFilename( splitNumber, isTraining=True, isLabel=False, testCluster=testCluster, extensionName='mat', isUsingMatlab=True )
    mlEstimatesMat = scipy.io.loadmat( trainingParamF )['alphaLearntM'] 
    mlEstimatesMat = np.matrix( mlEstimatesMat ) 
    mlEstimatesD = {}
    for topic in topicList:
        mlEstimatesD[topic] = np.array( mlEstimatesMat[ topic ] )
