# Using Mini-Batches for Dirichlet Compound Multinomial Language Model

Steps to replicate results for the twenty_newsgroups dataset.

DISCLAIMER: All the results reported are for twenty_newsgroups; the scripts have not been thoroughly tested for industry_sector dataset.

## 1. Run ./notebooks/data-extraction-and-cleaning.ipynb for data cleaning and extraction
 - creates three "intermediate" Python dictionary pickles that will be usd to split the corpus into training/testing matrices and label lists
 - the three pickled files should be moved into ./data_pickles
    * e.g.) ./data_pickles/twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle
 - see ./data_pickles/pickle_legend.txt for details on the three intermediate Python dictionary pickles

## 2. Running topic classification based on multinomial language model 
### Sci-kit Learn Multinomial NB: 
 - assumes ./notebooks/data-extraction-and-cleaning.ipynb has been run or that the ./data_pickles directory already contains the three "intermediate" pickled Python dictionaries
 - ./multinomial_twenty_newsgroups.py can be run as-is from the command line
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

DISCLAIMER: note that smoothing schemes between Python method and Matlab method are different.
For the Python classifier, alpha parameters are smoothed after **every** update iteration of the FPI method; for the Matlab classifier, alpha parameters are smoothed **once** after the FPI method has finished.
 
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

### Generating figures
#### Confusion matrices
 - ./multinomial_twenty_newsgroups.py, ./dirichletClassification.py, and ./matlabDirichletClassification.py all have the functions needed to plot confusion matrices and averaged confusion matrices
 
 #### Bubble plot
 - ./bubble_plotter.py can be run from the command line to generate an .html version of the bubble plot in the report
