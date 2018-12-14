import os, pickle
import numpy as np 
from random import sample
import time
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import itertools
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix, vstack# sparse matrix data structures
from sklearn import metrics
from statistics import mean, stdev
from sklearn.naive_bayes import MultinomialNB

def load_20newsgroups_lexicon(pickle_file_path='twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle', exclude_500_most_frequent=True):
    '''
    Loads a pickled dictionary of corpus-wide (token:corpus-wide frequency) pairs.
    Returns the list of lexicon tokens (w/o their frequency) sorted from most->least common

    Parameters:
        pickle_file_path: path to the pickled dictionary of corpus-wide (token:corpus-wide frequency)
            industry_sector_tokens_and_frequencies_across_dataset_list_of_tuples_(lexicon).pickle
            or
            twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle
    
    Returns:
        lexicon: the list of lexicon tokens sorted from most->least frequent in the corpus.
    '''
    # obtaining the lexicon (tokens) and sorting it by corpus-wide frequency    
    with open(pickle_file_path,'rb') as infile:
        corpus_wide_token_frequency_dict = pickle.load(infile)
    
    # sorts the tuples list by frequency most->least
    # corpus_wide_token_frequency_dict.sort(key=lambda x:x[1], reverse=True)
    
    # counting the number of tokens which only occur once in the entire corpus    
    '''unique_count = 0
    for (token,frequency) in token_and_frequency_tuple_list[::-1]:
        if frequency > 3:
            break
        else:
            unique_count += 1
    print(unique_count)'''

    lexicon = None
    if exclude_500_most_frequent:
        lexicon = [ token for (token,frequency) in list(corpus_wide_token_frequency_dict.items())[500:] ]
    else:
        lexicon = [ token for (token,frequency) in list(corpus_wide_token_frequency_dict.items()) ]
    return lexicon

def load_all_files_token_frequency_dictionaries(pickle_file_path):
    '''
    Loads and returns a pickled dictionary of dictionaries of the form

    docID_token_frequency_dictionaries = {
        docID#1 = {
            token#1:frequency in docID#1,
            token#2:frequency in docID#1,
            ...
        },
        docID#2 = {
            token#1:frequency in docID#2,
            token#2:frequency in docID#2,
            ...
        },
        ...

    }

    Parameters:
        pickle_file_path: path to the pickled dictionary of dictionaries.
        'twenty_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'
        or
        'sector_dict_of_dicts_filename_token_freq_dicts.pickle'

    Returns:
        the unpickled dictionary of dictionaries
    '''    

    with open(pickle_file_path,'rb') as infile:
        docID_token_frequency_dictionaries = pickle.load(infile)
    # recall that docID_token_frequency_dictionaries has the form
    # where docID# is the file name 
    '''
        docID_token_frequency_dictionaries = {
            docID#1 = {
                token#1:frequency in docID#1,
                token#2:frequency in docID#1,
                ...
            },
            docID#2 = {
                token#1:frequency in docID#2,
                token#2:frequency in docID#2,
                ...
            },
            ...

        }
    '''
    return docID_token_frequency_dictionaries
 
def setup(path_to_pickle_directory, iteration, training_proportion=0.8):
    '''
    Loads the data pickles to create and populate a sparse array whose dimensions are 
    # files x # tokens (corpus-wide). This sparse array is randomly subdivided
    <iteration> number of times to create <iteration> different training and testing submatrices (the
    topic classes are equally represented in both submatrices).

    Arguments:

        path_to_pickle_directory: path to the directory where the pickled data files are saved.
        
        iteration: integer representing the number of cross-validations to prepare.

        training_proportion: float between 0.0 and 1.0 indicating the proportion of the data to be kept for training.

    Returns:

        Nothing, but creates <iteration> groups of 4 pickled files: 
            
            twenty_newsgroups_testing_matrix_labels_vector_cv#_.pickle,
            twenty_newsgroups_testing_matrix_cv#_.pickle,
            twenty_newsgroups_training_matrix_labels_vector_cv#_.pickle,
            twenty_newsgroups_training_matrix_cv#_.pickle
    '''
    assert (os.path.isdir(path_to_pickle_directory))
    assert (iteration > 0)
    assert (0.0 < training_proportion < 1.0)

    # 1. loading lexicon
    # exclude_500_most_frequent is set to False because there's a more efficient way to deal with this stuff
    path_to_lexicon_containing_file = os.path.join(path_to_pickle_directory, 'twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle')
    lexicon = load_20newsgroups_lexicon(path_to_lexicon_containing_file, exclude_500_most_frequent=False)
    
    # 2. making a lexicon:index dictionary for quick index-finding
    token_lexicon_index_dict = {  }
    for index, token in enumerate(lexicon):
        token_lexicon_index_dict[token] = index

    # 3. loading data
    path_to_data = os.path.join(path_to_pickle_directory, "twenty_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle")
    dict_of_filename_to_dict_of_token_to_frequency_pairs = load_all_files_token_frequency_dictionaries(path_to_data)
    '''
    dict_of_filename_to_dict_of_token_to_frequency_pairs has the following layout:

        dict_of_filename_to_dict_of_token_to_frequency_pairs = {
            file #1's name = {
                token#1:frequency in docID#1,
                token#2:frequency in docID#1,
                ...
            },
            file #2's name = {
                token#1:frequency in docID#2,
                token#2:frequency in docID#2,
                ...
            },
            ...
    
    Note: the 'file # name' keys are formatted as "class_name+document_ID" (e.g. alt.atheism+49960)
    '''

    # 4. create auxiliary dictionaries to map from 
    # topic name -> topic ID (int),
    # filename -> topic ID (int),
    # topic ID (int) -> number of files belonging to this topic
    
    file_name_to_topicINT_dict = {} # (file's name:integer representing that file's topic) (key:value) pairs
    topic_INTs_and_frequencies_dict = {} # (integer representing a topic:number of files in that topic) (key:value) pairs
    topicID_to_topic_INT_mapping = {} # maps from topic name -> the integer label for that topic name
    
    for filename_key in dict_of_filename_to_dict_of_token_to_frequency_pairs.keys():
        assert '+' in filename_key
        topicID, _ = filename_key.split('+')
        
        if topicID in topicID_to_topic_INT_mapping.keys():
            file_name_to_topicINT_dict[ filename_key ] = topicID_to_topic_INT_mapping[ topicID ]
            topic_INTs_and_frequencies_dict[ topicID_to_topic_INT_mapping[ topicID ] ] += 1
        
        else:
            topicID_to_topic_INT_mapping[topicID] = len(topicID_to_topic_INT_mapping) + 1
            topic_INTs_and_frequencies_dict[ topicID_to_topic_INT_mapping[ topicID ] ] = 1
            file_name_to_topicINT_dict[filename_key] = topicID_to_topic_INT_mapping[ topicID ]
    
    assert (len(topic_INTs_and_frequencies_dict) == 20)
    assert (len(topicID_to_topic_INT_mapping) == 20)
    

    # 5. initializing a single sparse matrix
    sparse_matrix = dok_matrix(
        (
            len( list( dict_of_filename_to_dict_of_token_to_frequency_pairs.keys() ) ), # equivalent to number of documents in corpus
            (len( lexicon ) + 1) # the extra column (last one) will contain the class label
        ),
        dtype=np.float32
    )

    # 6. populating matrix with frequencies
    
    for document_row_index, (file_name, token_frequency_dict) in enumerate( dict_of_filename_to_dict_of_token_to_frequency_pairs.items() ):
        print(f"{document_row_index+1} / {len(dict_of_filename_to_dict_of_token_to_frequency_pairs)}")
        for token, frequency in token_frequency_dict.items():
            sparse_matrix[ document_row_index, token_lexicon_index_dict[ token ] ] = frequency
            #print(sparse_matrix[document_row_index, token_lexicon_index_dict[token]])
        sparse_matrix[ document_row_index, -1 ] = int( file_name_to_topicINT_dict[file_name] )# sets this document's feature vector's last entry to be the class label

    # 7. making a list of the 500 most frequent columns' indices (representing tokens)

    column_sums = sparse_matrix.sum(axis=0).flatten().tolist()[0][:-1] # we don't want to include the class label column in these operations
    assert len(column_sums) == len(lexicon)

    column_indices_and_sums = list( 
        zip(
            list( range( len( column_sums ) ) ),
            column_sums
        )
    )

    column_indices_and_sums.sort(key=lambda x:x[1], reverse=True)
    column_indices_to_remove = [ index for (index, sum) in column_indices_and_sums[0:500] ]
    column_indices_to_keep = [ index for index in range(len(lexicon) + 1) if index not in column_indices_to_remove ]

    # 8. making a new sparse matrix keeping only the columns correspondings to tokens which weren't the 500 most frequent tokens
    sparse_csc_matrix = csc_matrix(sparse_matrix)
    sparse_matrix_wo_500_most_frequent_tokens = sparse_csc_matrix[:,column_indices_to_keep]

    # 9. making the iteration train/test 50/50 splits.

    for c in range(iteration):

        this_iteration_training_matrix, this_iteration_testing_matrix = None, None 
        # csr_matrix((1,sparse_matrix_wo_500_most_frequent_tokens.shape[1])), 
        this_iteration_training_labels, this_iteration_testing_labels = [],[]

        print(f"iteration {c+1} of {iteration}")
        for class_index_in_iteration, class_label in enumerate( list( topic_INTs_and_frequencies_dict.keys() ) ): 

            number_of_class_instances = topic_INTs_and_frequencies_dict[ class_label ]
            row_indices_of_class_instances_in_matrix = [ row_index for row_index in range(sparse_matrix_wo_500_most_frequent_tokens.shape[0]) if sparse_matrix_wo_500_most_frequent_tokens[row_index, -1] == class_label ]
            
            assert(number_of_class_instances == len(row_indices_of_class_instances_in_matrix))
            
            # samples w/o replacement from the row_indices_of_class_instances_in_matrix
            list_of_row_indices_for_training = sample(row_indices_of_class_instances_in_matrix, int( training_proportion *len(row_indices_of_class_instances_in_matrix)) ) # parametize
            list_of_row_indices_for_testing = [ row_index for row_index in row_indices_of_class_instances_in_matrix if row_index not in list_of_row_indices_for_training ]

            print(f"{len(list_of_row_indices_for_training)} class {class_label} documents for training, {len(list_of_row_indices_for_testing)} for testing (out of {number_of_class_instances}).")

            # compiles this class's training row vectors (except for their last entry) in a new array
            this_cv_training_matrix = sparse_matrix_wo_500_most_frequent_tokens[list_of_row_indices_for_training,:-1]
            this_cv_training_labels = sparse_matrix_wo_500_most_frequent_tokens[list_of_row_indices_for_training,-1].data.tolist()
           
            # updates the overall training matrix
            if class_index_in_iteration == 0:
                this_iteration_training_matrix = this_cv_training_matrix
            else:
                this_iteration_training_matrix = vstack([this_iteration_training_matrix,this_cv_training_matrix])

            assert (len(this_cv_training_labels) == this_cv_training_matrix.shape[0])
            this_iteration_training_labels.extend(this_cv_training_labels)

            this_cv_testing_matrix = sparse_matrix_wo_500_most_frequent_tokens[list_of_row_indices_for_testing,:-1]
            this_cv_testing_labels = sparse_matrix_wo_500_most_frequent_tokens[list_of_row_indices_for_testing,-1].data.flatten().tolist()
            
            if class_index_in_iteration == 0:
                this_iteration_testing_matrix = this_cv_testing_matrix
            else:
                this_iteration_testing_matrix = vstack([this_iteration_testing_matrix,this_cv_testing_matrix])

            assert (len(this_cv_testing_labels) == this_cv_testing_matrix.shape[0])
            this_iteration_testing_labels.extend(this_cv_testing_labels)

            print(f"shape of training matrix so far: {this_iteration_training_matrix.shape}, length of training labels list: {len(this_iteration_training_labels)}")
            print(f"shape of testing matrix so far: {this_iteration_testing_matrix.shape}, length of testing labels list: {len(this_iteration_testing_labels)}")

        assert (len(this_iteration_testing_labels) == this_iteration_testing_matrix.shape[0])
        assert (len(this_iteration_training_labels) == this_iteration_training_matrix.shape[0])

        with open('new_twenty_newsgroups_training_matrix_cv#{}.pickle'.format(str(c)), 'wb') as handle:
            pickle.dump(this_iteration_training_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('new_twenty_newsgroups_training_matrix_labels_vector_cv#{}.pickle'.format(str(c)), 'wb') as handle:
            pickle.dump(this_iteration_training_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('new_twenty_newsgroups_testing_matrix_cv#{}.pickle'.format(str(c)), 'wb') as handle:
            pickle.dump(this_iteration_testing_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('new_twenty_newsgroups_testing_matrix_labels_vector_cv#{}.pickle'.format(str(c)), 'wb') as handle:
            pickle.dump(this_iteration_testing_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n\nFinished making the training/testing submatrices and label lists.\n\nPlease move them into a directory following this path:\n'./training_testing_and_results_pickles/20_newsgroups/cv# \naccording to their cv # tag.")
    raise SystemExit

def run_multinomialNB(training_matrix_pickle, training_labels_pickle, testing_matrix_pickle, testing_labels_pickle):
    '''
    Trains a multinomial Naive-Bayes classifier using the training data submatrix and labels.
    The classifier is then evaluated on the testing pickle files.
    The classifier itself and the resulting confusion matrix are saved as pickles.

    Arguments:
        
        training_matrix_pickle: path to the training matrix pickled file.

        training_labels_pickle: path to the training labels pickled file.

        testing_matrix_pickle: path to the testing matrix pickled file.

        testing_labels_pickle: path to the testing labels pickled file.

    Returns:

        Nothing, but the classifier and its confusion matrix are pickled in the current directory.

    ''' 
    # recover pickles
    with open(training_matrix_pickle,'rb') as handle:
        training_matrix = pickle.load(handle)
    
    with open(training_labels_pickle, 'rb') as handle:
        training_labels = pickle.load(handle)
    
    with open(testing_matrix_pickle,'rb') as handle:
        testing_matrix = pickle.load(handle)

    with open(testing_labels_pickle, 'rb') as handle:
        testing_labels = pickle.load(handle)

    print(f"shape of training matrix = {training_matrix.shape}, shape of training labels list = {len(training_labels)}")
    print(f"testing labels: {set(testing_labels)}")
    print(f"training labels: {set(training_labels)}")
   
    # don't forget to permute the training matrix and labels together!

    try:
        assert training_matrix.shape[0] == len(training_labels)
    except AssertionError:
        print(f"shape of training matrix = {training_matrix.shape}, length of training labels list = {len(training_labels)}")

    mnnb_classifier = MultinomialNB(alpha=0.01,fit_prior=False) # to match the Madsen et al.'s multinomial classifier's parameters
    startTime = time.time()
    mnnb_classifier.fit(training_matrix, training_labels)
    endTime = time.time()

    print(f"Trained the classifier in {endTime - startTime} seconds")

    predictions = mnnb_classifier.predict(testing_matrix)
    print(predictions.shape)

    confusion_matrix_unnormalized = metrics.confusion_matrix(testing_labels, predictions, labels=list(set(testing_labels)))
    confusion_matrix = confusion_matrix_unnormalized.astype('float') / (confusion_matrix_unnormalized.sum(axis=1)[:, np.newaxis])
    print(f"confusion matrix shape: {confusion_matrix.shape}")
    
    # need to save the confusion matrix and the classifier itself
    # saves the classifier and confusion matrix in the appropriate cv# folder/directory

    with open(os.path.join(os.path.dirname(training_matrix_pickle), 'trained_multinomialNB_classifier.pickle'), 'wb') as handle:
        pickle.dump(mnnb_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(os.path.dirname(training_matrix_pickle), 'confusion_matrix.pickle'), 'wb') as handle:
        pickle.dump(confusion_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_average_confusion_matrix(cv_directory, number_of_cvs=10):
    '''
    Compiles the <number_of_cvs> confusion matrices in to a single, averaged confusion matrix.

    Arguments: 

        cv_directory: path to the ...

        number_of_cvs: integer representing the number of confusion matrices to look for and compile.

    Returns: 

        Nothing, but generates a plot of the averaged confusion matrix and prints the 
        <number_of_cvs> classifiers' average precision and standard deviation.
        
    '''
    assert os.path.isdir(cv_directory)
    print("generating average confusion matrix")
    confusion_matrices_list = []
    for cv in range(number_of_cvs):
        with open(os.path.join(cv_directory, f"cv{cv}", "confusion_matrix.pickle"), 'rb') as cm_handle:
            confusion_matrices_list.append( pickle.load( cm_handle ) )

    # making sure all confusion matrices have the same shape
    for index in range(1,number_of_cvs):
        assert confusion_matrices_list[index].shape == confusion_matrices_list[index-1].shape
    
    average_confusion_matrix = np.zeros((confusion_matrices_list[0].shape))

    for e,matrix in enumerate(confusion_matrices_list):
        print(e)
        print(matrix.shape)
        for row_index, row in enumerate(matrix):
            for col_index, entry in enumerate(row):
                average_confusion_matrix[row_index, col_index] += float( entry / ( number_of_cvs ) )*100

    print(average_confusion_matrix)

    topics = [ str(i) for i in range(average_confusion_matrix.shape[0]) ]

    # plotting the average confusion matrix

    fig = plt.figure()
    ax = fig.subplots(nrows=1, ncols=1)
    heatmap = ax.matshow(average_confusion_matrix, cmap="Reds", aspect="equal")

    # setting up axes 
    ax.set_xticks(np.arange(len(average_confusion_matrix)))
    ax.set_yticks(np.arange(len(average_confusion_matrix)))
    ax.set_xticklabels(topics)
    ax.set_yticklabels(topics)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", va="top",
         rotation_mode="anchor")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar(heatmap)
    # setting up labels / annotations
    '''matplotlib.rcParams.update({'font.size': 6})
    for col_index in range(len(average_confusion_matrix)):
        for row_index in range(len(average_confusion_matrix)):
            text = ax.text(row_index, col_index, 
                            '{0:.1f}'.format(average_confusion_matrix[col_index, row_index]), # 2 decimal places
                            ha="center", va="center", color="grey"
                        )
    '''
    plt.tight_layout()
    plt.show()

    precision_over_cvs = [0]*number_of_cvs # list which will store each cv's precision

    for cv_index, cm in enumerate(confusion_matrices_list):
        this_cv_tps, this_cv_fps = 0, 0

        # iterate over classes to find each class's true positives and false positives for this cv
        for class_index, class_row in enumerate(cm):
            tp = cm[class_index, class_index] # true positives = score at the diagonal
            fp = sum(cm[:,class_index]) - tp # false positives = sum of the class_index column - true positives
            this_cv_tps += tp
            this_cv_fps += fp
        
        precision_over_cvs[cv_index] = this_cv_tps/(this_cv_tps+this_cv_fps)
    
    print(f"Precision: {mean(precision_over_cvs)} +/- {stdev(precision_over_cvs)}")
  

if __name__ == '__main__':

    # uncomment the line below to generate 10 different training/testing matrix and label lists pairs
    #setup('\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'data_pickles')), 10, training_proportion=0.8)
    
    # The following loop evaluates the 10 multinomialNB classifiers on their respective testing set,
    # producing a confusion matrix for each of the 10 cross-validations.
    for k in range(0,10):
        print(f"iteration {k}")
        startTime = time.time()
        training_testing_and_results_pickles_path = '\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'training_testing_and_results_pickles', '20_newsgroups', 'cv{}'.format(str(k))))
        
        run_multinomialNB(
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_training_matrix_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_training_matrix_labels_vector_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_testing_matrix_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_testing_matrix_labels_vector_cv#{}.pickle'.format(str(k)))
        )
    
    # The following generates an averaged confusion matrix from the 10 cross-validation confusion matrices. 
    path_to_cv_directory = '\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'training_testing_and_results_pickles', '20_newsgroups_new'))
    generate_average_confusion_matrix(path_to_cv_directory)