# Using Mini-Batches for Dirichlet Compound Multinomial Language Model

Steps to replicate results for the twenty_newsgroups dataset.

DISCLAIMER: All the results reported are for twenty_newsgroups; the scripts have not been thoroughly tested for industry_sector dataset.

## 1. Run jupyter-notebook for data pre-processing and 80-20 cross-validation splits
 - creates pickles of training data, training labels, testing data and testing labels
    * e.g.) path to cross-validation 0 trial for twenty_newsgroups: 
          <dataDir>/twenty_newsgroups/cv0/twenty_newsgroups_*.pickle

## 2. Running topic classification based on multinomial language model 
### Sci-kit Learn Multinomial NB: 
 - *.py
 - this is the implementation reported in the report
 

### Multinomial NB from scratch:
 - this is NOT reported in the report; multinomialClassification.py shares most of code with dirichletClassification.py 
 - multinomialClassification.py numTrials TEST_CLUSTER
   * numTrials is a number from 1-10, specifying number of cross-validation trials included for classification
   * TEST_CLUSTER is a string, either "twenty_newsgroups" or "industry_sector" (quotations aren't needed for use in command line)
 - output of running the script:
   * time taken to run predictions are reported
   * confusion matrix for classification
   * precision result for classification

## 3. Running topic classification based on dirichlet compound multinomial language model
### Python method: 
 - dirichletClassification.py maxIter numDocsPerIter numTrials thresholdPower TEST_CLUSTER
   * maxIter is a positive integer
   * numDocsPerIter: specifies whether to use mini-batch (1~798) or GD (799) for twenty_newsgroups
   * numTrials: number of CV trials for classification
   * thresholdPower: is a number specifying training threshold for parameter approximation. Training Threshold will be e-(thresholdPower)
   * TEST_CLUSTER: same as in multinomialClassification.py. Either "twenty_newsgroups" or "industry_sector"
 - output of running the script:
   * various prints on training time, number of documents used per topic
   * confusion matrix for classification
   * precision result for classification
 
### Matlab-based method: 
We were interested to offload heavy computation to [Minka's fastfit matlab implementation](https://github.com/tminka/fastfit), as the pure python implementation was taking a long time to train on personal laptops. We were unable to automate this offloading using matlab.engine from python (that enables using Matlab functions in python).

To run the Matlab-based method, all you need is a working version of Matlab (tested on Matlab R2016b, MacOS), and Python.
The _**matlab/**_ directory is a combination of [Minka's fastfit](https://github.com/tminka/fastfit) and dependencies from [Minka's lightspeed](https://github.com/tminka/lightspeed).

 As it is now, 3 things have to be done manualy to offload heavy-computation (trainParameter) to matlab. 
 - transfer python pickle cross-validation splits to .mat file
   * e.g.) computeInMatlab.py TEST_CLUSTER cvTrialNumber
   * generates training data, ending in *training_matrix_cv#cvTrialNumber.mat
 - open Matlab, and run learnParametersScript.m
   * generates a pickle file called *trained_param_cv#cvTrialNumber.pickle
   * for multiple CV splits, computeInMatlab.py and learntParameters.Script.m need to be manually modified
 - run matlabDirichletClassification.py numTrials smooth TEST_CLUSTER
    * different from dirichletClassification.py in only offloading parameter training 

