% script to learn alphas
totalSplitNum = 1;

testCluster = 'twenty_newsgroups';
isTwentyNewsgroup = strcmp( testCluster, 'twenty_newsgroups' );
numTopics = 20;
dataDir = '../../training_testing_and_results_pickles/';
fNameFormat = strcat( dataDir, testCluster, '/cv%d/', testCluster, '_training_matrix_cv#%d.mat' );
learntParamFormat = strcat( dataDir, testCluster, '/cv%d/', testCluster, '_trained_param_cv#%d.mat' );
for splitIdx = 1:totalSplitNum
    splitIdx = splitIdx - 1; 
    fName = sprintf( fNameFormat, splitIdx, splitIdx );
    learntFName = sprintf( learntParamFormat, splitIdx, splitIdx );
    trainingData = load( fName );
    % fields are 'arr' + topic#
    
    % generate initial alpha, and variable-indexable name for imported .mat
    % file
    sz = size( trainingData.arr0 );
    alphaInit = ones( 1, sz(2) );
    topicDataMFns = fieldnames( trainingData );
    
    alphaLearntM = zeros( numTopics, sz(2) );
    for topicIdx = 1:numTopics
        if topicIdx == 17
            continue;
        end
        trainingDataM = trainingData.( topicDataMFns{ topicIdx } );
        alphaLearntM( topicIdx, : ) = polya_fit_simple( trainingDataM, alphaInit );
    end
    save( learntFName, 'alphaLearntM' );
        
end