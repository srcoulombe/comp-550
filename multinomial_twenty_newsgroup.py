import os, pickle
import numpy as np 
from random import sample
from scipy.sparse import dok_matrix # sparse matrix data structure
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import itertools
from statistics import mean, stdev

def load_20newsgroups_lexicon(pickle_file_path, exclude_500_most_frequent=True):
    '''
    Loads a pickled dictionary of corpus-wide (token:corpus-wide frequency) (key:value) pairs.
    Returns the list of lexicon tokens (w/o their frequency) sorted from most->least common

    Parameters:
        pickle_file_path: path to the pickled dictionary of corpus-wide (token:corpus-wide frequency)
            twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle 
    
    Returns:
        lexicon: the list of lexicon tokens sorted from most->least frequent in the corpus.
    '''
    # obtaining the lexicon (tokens) and sorting it by corpus-wide frequency    
    with open(pickle_file_path,'rb') as infile:
        corpus_wide_token_frequency_dict = pickle.load(infile)
    
    # sorts the dictionary by value
    token_and_frequency_tuple_list = sorted(corpus_wide_token_frequency_dict.items(), key=lambda kv: kv[1]) 
    
    # reorders tuple list from most->least token_and_frequency_tuple_list
    token_and_frequency_tuple_list.reverse() 

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
        lexicon = [ token for (token,frequency) in token_and_frequency_tuple_list[500:] ] # excludes the top 500 most common tokens
    else:
        lexicon = [ token for (token,frequency) in token_and_frequency_tuple_list ]
    return lexicon

def classify_topics(pickle_file_path):
    '''
    Loads a pickled dictionary of (topic_name:list of files with this topic) (key:value) pairs,
    and returns two dictionaries:
    dictionary #1 (class_to_topic_mapping): an (int [0-19]) : (string topic) dictionary.
    dictionary #2 (class_to_documents_dictionary): an (int [0-19]) : list of document names (file names) dictionary.

    Note: if you are using this with the '!industry_sector_topicID_to_list_of_topical_files_dictionary.pickle' file,
    the topics are integers from [100,203]. Their corresponding topic names can be found in the 
    !industry_sector_subtopic_key_to_subtopic_name_dictionary.pickle file
    which is a dictionary of string-representations of integers in ([100,203]:topic name) (key:value) pairs
    (e.g. '100':'sector\\basic.materials.sector\\chemical.manufacturing.industry')

    Parameters:
        pickle_file_path: path to the pickled dictionary of (topic_name:list of files with this topic) (key:value) pairs.
        'twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'
        or
        '!industry_sector_topicID_to_list_of_topical_files_dictionary.pickle'

    Returns:
        dictionary #1 (class_to_topic_mapping): an (int [0-19]) : (string topic) dictionary.
        dictionary #2 (class_to_documents_dictionary): an (int [0-19]) : list of document names (file names) dictionary.
    '''
    with open(pickle_file_path,'rb') as infile:
        topic_to_topical_document_IDs_dict = pickle.load(infile)
    
    class_to_topic_mapping = {} # a (int [0-19]) : (string topic) dictionary
    class_to_documents_dictionary = {} # a (int [0-19]) : list of document IDs dictionary
    for class_int, (topic, filename_and_valid_lines_dictionary) in enumerate(list(topic_to_topical_document_IDs_dict.items())):

        class_to_topic_mapping[class_int] = topic
        class_to_documents_dictionary[class_int] = list(filename_and_valid_lines_dictionary.keys())

    return class_to_topic_mapping, class_to_documents_dictionary

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
        '!sector_dict_of_dicts_filename_token_freq_dicts.pickle'

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
    
def get_training_testing_split(topic_class_to_list_of_topical_documents_dictionary, training_proportion=0.8):
    '''
    Splits the documents in the input (topic class integer:list of files with this topic) dictionary into training/testing dictionaries
    according to the training_proportion parameter while ensuring an equal inclusion of documents in the training list
    over all topic classes (so that training_proportion of class #1 is in the training dictionary, and training_proportion of class #2 is in the training
    dictionary, and training_proportion of class #3 is in the training dictionary, etc...).

    Parameters:
        topic_class_to_list_of_topical_documents_dictionary: the dictionary of (topic class integer:list of files with this topic) obtained
            as classify_topics()'s second return value.

        training_proportion: float between 0.0 and 1.0 indicating the proportion of documents to include in the training set.

    Returns:
        training_set_document_IDs: dictionary of (topic class integer:list of files in this class included in the training set) (key:value) pairs.
        
        testing_set_document_IDs = dictionary of (topic class integer:list of files in this class included in the testing set) (key:value) pairs.
    '''
    training_set_document_IDs = {}
    testing_set_document_IDs = {}

    for class_key, list_of_files_in_this_topic_class in topic_class_to_list_of_topical_documents_dictionary.items():

        number_of_documents_in_training_set = int(training_proportion*len(list_of_files_in_this_topic_class)) 

        # samples w/o replacement from the list_of_files_in_this_topic_class
        list_of_document_in_training_set = sample(list_of_files_in_this_topic_class, number_of_documents_in_training_set)

        # getting leftover documents
        list_of_document_in_testing_set = [ doc_name for doc_name in list_of_files_in_this_topic_class if doc_name not in list_of_document_in_training_set ]

        reconstructed_list_of_files_in_this_topic_class = list_of_document_in_training_set + list_of_document_in_testing_set
        # sanity check
        assert (len(reconstructed_list_of_files_in_this_topic_class) == len(list_of_files_in_this_topic_class))
        
        training_set_document_IDs[class_key] = list_of_document_in_training_set
        testing_set_document_IDs[class_key] = list_of_document_in_testing_set
    
    return training_set_document_IDs, testing_set_document_IDs

def setup(path_to_pickle_directory, iteration=None):

# obtaining the lexicon (tokens) and sorting it by corpus-wide frequency    
    lexicon = load_20newsgroups_lexicon(os.path.join(path_to_pickle_directory, 'twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle'))

# mapping the 20 topics to integer classes [0-19]
    class_to_topic_mapping, class_to_documents_dictionary = classify_topics(os.path.join(path_to_pickle_directory, 'twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'))

# loading the token-frequency dictionaries for all files
    docID_token_frequency_dictionaries = load_all_files_token_frequency_dictionaries(os.path.join(path_to_pickle_directory, 'twenty_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'))
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

# 1. splitting data into training/testing
# this could be a one-liner using sklearn.model_selection.train_test_split, but we just need to ensure an equal sampling across each class...
    
    training_set_document_IDs, testing_set_document_IDs = get_training_testing_split(class_to_documents_dictionary)
    number_of_training_documents = sum( [ len(list_of_training_documents_for_this_class) for list_of_training_documents_for_this_class in training_set_document_IDs.values() ] )
    number_of_testing_documents = sum( [ len(list_of_testing_documents_for_this_class) for list_of_testing_documents_for_this_class in testing_set_document_IDs.values() ] ) 

# 2. making the sparse training and testing matrices

    training_sparse_matrix = dok_matrix(
        ( 
            number_of_training_documents,
            len(lexicon)
        ), 
        dtype=np.float32
    )

    testing_sparse_matrix = dok_matrix(
        ( 
            number_of_testing_documents,
            len(lexicon)
        ), 
        dtype=np.float32
    )

# 3. populating the sparse training matrix

    training_labels_vector = []       
    this_document_row_index_in_matrix = 0
    
    # for every document_list in training_set_document_IDs,
    # for every document in document_list,
    # for every token in document,
    # update training_sparse_matrix[this document's row][this token's column] = token's frequency in this document
    # add the document's topic class (an integer) to the training_labels_vector (a list)

    for class_integer, class_document_list in training_set_document_IDs.items(): # could include a column for the class integer label
        
        print(f"Class: {class_integer} : {class_to_topic_mapping[class_integer]}")

        for document_row, document in enumerate(class_document_list):
            
            # updating the label vector/list
            training_labels_vector.append(class_integer)
            
            # uncomment the following line if you just want to get the labels 
            # continue 

            this_documents_token_frequency_dict = docID_token_frequency_dictionaries[document] 

            # finding the sublist of the tokens in this document which are in 'lexicon'.
            # this is important for when we exclude the 500 most frequent words from the lexicon,
            # as we would otherwise have (token:frequency) (key:value) pairs in this_documents_token_frequency_dict
            # which do not correspond to entries in 'lexicon'.
            
            # real list
            lexicon_indices_of_tokens_to_consider_in_this_document = []
            
            # the following list is for sanity checks only, will be commented out afterwards
            # can be commented out because an entire class's documents fulfilled the sanity checks
            # that the following list was made for (and those sanity checks are long)
            tokens_to_consider_in_this_document = []
            
            for token in list(this_documents_token_frequency_dict.keys()):
                try:
                    lexicon_indices_of_tokens_to_consider_in_this_document.append(lexicon.index(token))
                    # can be commented out
                    tokens_to_consider_in_this_document.append(token)
                except ValueError:
                    continue

            # sanity check
            # can be commented out because an entire class's documents fulfilled the sanity checks
            # and the sanity check is slow
            assert len(tokens_to_consider_in_this_document) == len(lexicon_indices_of_tokens_to_consider_in_this_document)

            # making a sub-dictionary of (token's index in the lexicon:frequency) (key:value) pairs for the tokens in this document which aren't in the 500 most frequent tokens list
            this_documents_token_index_frequency_dict_wo_500_most_frequent_words = {key: this_documents_token_frequency_dict[ lexicon[ key ] ]  for key in lexicon_indices_of_tokens_to_consider_in_this_document}
            
            # sanity checks
            # again, can be commented out because they seemed successful and they are slow
            this_documents_token_index_frequency_dict_wo_500_most_frequent_words_keys = this_documents_token_index_frequency_dict_wo_500_most_frequent_words.keys()
            #for token_to_consider, token_index_in_lexicon in zip(tokens_to_consider_in_this_document, lexicon_indices_of_tokens_to_consider_in_this_document):
                #assert token_to_consider == lexicon[ token_index_in_lexicon ]

            for token_index, freq in this_documents_token_index_frequency_dict_wo_500_most_frequent_words.items():
                
                training_sparse_matrix[this_document_row_index_in_matrix, token_index] = freq
            
            this_document_row_index_in_matrix += 1
        
            print(f"{this_document_row_index_in_matrix} / {len(class_document_list)}")
        
        print(f"Class: {class_integer}, document_row = {document_row}, finished appending {training_labels_vector[-1]} to the training labels vector")
    
    # sanity check
    #assert ((this_document_row_index_in_matrix) == number_of_training_documents)

    print(f"{number_of_training_documents} training documents, size of label array = {len(training_labels_vector)}")

    if iteration is None:

        with open('twenty_newsgroups_training_matrix.pickle','wb') as handle:
            pickle.dump(training_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('twenty_newsgroups_training_matrix_labels_vector.pickle','wb') as handle:
            pickle.dump(training_labels_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:

        with open('twenty_newsgroups_training_matrix_cv#{}.pickle'.format(str(iteration)),'wb') as handle:
            pickle.dump(training_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('twenty_newsgroups_training_matrix_labels_vector_cv#{}.pickle'.format(str(iteration)),'wb') as handle:
            pickle.dump(training_labels_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    print("done populating the training matrix")

# 4. populating the sparse testing matrix
    
    testing_labels_vector = []
    this_document_row_index_in_matrix = 0

    for class_integer, class_document_list in testing_set_document_IDs.items():
        
        print(f"Class: {class_integer}: {class_to_topic_mapping[class_integer]}")

        for document_row, document in enumerate(class_document_list):
            
            # updating the label vector/list

            testing_labels_vector.append(class_integer)
            # uncomment the following line if you just want to get the labels 
            # continue

            this_documents_token_frequency_dict = docID_token_frequency_dictionaries[document] 

            # finding the sublist of the tokens in this document which are in 'lexicon'.
            # this is important for when we exclude the 500 most frequent words from the lexicon,
            # as we would otherwise have (token:frequency) (key:value) pairs in this_documents_token_frequency_dict
            # which do not correspond to entries in 'lexicon'.
            
            # real list
            lexicon_indices_of_tokens_to_consider_in_this_document = []
            
            # the following list is for sanity checks only, will be commented out afterwards
            # can be commented out because an entire class's documents fulfilled the sanity checks
            # that the following list was made for (and those sanity checks are long)
            tokens_to_consider_in_this_document = []

            for token in list(this_documents_token_frequency_dict.keys()):
                try:
                    lexicon_indices_of_tokens_to_consider_in_this_document.append(lexicon.index(token))
                    # can be commented out
                    tokens_to_consider_in_this_document.append(token)
                except ValueError:
                    continue

            # sanity check
            # can be commented out because an entire class's documents fulfilled the sanity checks
            # and the sanity check is slow
            assert len(tokens_to_consider_in_this_document) == len(lexicon_indices_of_tokens_to_consider_in_this_document)

            # making a sub-dictionary of (token's index in the lexicon:frequency) (key:value) pairs for the tokens in this document which aren't in the 500 most frequent tokens list
            this_documents_token_index_frequency_dict_wo_500_most_frequent_words = {key: this_documents_token_frequency_dict[ lexicon[ key ] ] for key in lexicon_indices_of_tokens_to_consider_in_this_document}
            
            # sanity checks
            # again, can be commented out because they seemed successful and they are slow
            this_documents_token_frequency_dict_wo_500_most_frequent_words_keys = this_documents_token_index_frequency_dict_wo_500_most_frequent_words.keys()
            #for token_to_consider, token_index_in_lexicon in zip(tokens_to_consider_in_this_document, lexicon_indices_of_tokens_to_consider_in_this_document):
                #assert token_to_consider == lexicon[ token_index_in_lexicon ]

            for token_index, freq in this_documents_token_index_frequency_dict_wo_500_most_frequent_words.items():

                testing_sparse_matrix[this_document_row_index_in_matrix, token_index] = freq

            this_document_row_index_in_matrix += 1
        
        print(f"Class: {class_integer}, document_row = {document_row}, finished appending {testing_labels_vector[-1]} to the testing labels vector")

    #assert ((this_document_row_index_in_matrix) == number_of_testing_documents)
    #input(f"{number_of_testing_documents} testing documents, size of label array = {len(testing_labels_vector)}")
    
    if iteration is None:

        with open('twenty_newsgroups_testing_matrix.pickle','wb') as handle:
            pickle.dump(testing_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('twenty_newsgroups_testing_matrix_labels_vector.pickle','wb') as handle:
            pickle.dump(testing_labels_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open('twenty_newsgroups_testing_matrix_cv#{}.pickle'.format(str(iteration)),'wb') as handle:
            pickle.dump(testing_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('twenty_newsgroups_testing_matrix_labels_vector_cv#{}.pickle'.format(str(iteration)),'wb') as handle:
            pickle.dump(testing_labels_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)        

    print("done populating the test matrix")
    
def run_multinomialNB(training_matrix_pickle, training_labels_pickle, testing_matrix_pickle, testing_labels_pickle):
    
    # recover pickles
    with open(training_matrix_pickle,'rb') as handle:
        training_matrix = pickle.load(handle)
    
    with open(training_labels_pickle, 'rb') as handle:
        training_labels = pickle.load(handle)
    
    with open(testing_matrix_pickle,'rb') as handle:
        testing_matrix = pickle.load(handle)

    with open(testing_labels_pickle, 'rb') as handle:
        testing_labels = pickle.load(handle)
            
    print(f"length of training matrix = {training_matrix.shape[0]}, length of training labels row matrix = {len(training_labels)}")
    print(f"testing labels: {list(set(testing_labels))}")
    print(f"training labels: {list(set(training_labels))}")
    testlabels = Counter(testing_labels)
    trainlabels = Counter(training_labels)
    # don't forget to permute the training matrix and labels together!

    try:
        assert training_matrix.shape[0] == len(training_labels)
    except AssertionError:
        print(f"length of training matrix = {len(training_matrix)}, length of training labels row matrix = {len(training_labels)}")

    mnnb_classifier = MultinomialNB(alpha=0.01,fit_prior=False) # to match the paper's multinomial classifier's parameters
    mnnb_classifier.fit(training_matrix, training_labels)
    print("done training the classifier")

    predictions = mnnb_classifier.predict(testing_matrix)
    print(predictions.shape)

    confusion_matrix_unnormalized = metrics.confusion_matrix(testing_labels, predictions, labels=list(set(testing_labels)))
    confusion_matrix = confusion_matrix_unnormalized.astype('float') / confusion_matrix_unnormalized.sum(axis=1)[:, np.newaxis]
    # need to save the confusion matrix and the classifier itself
    # saves the classifier and confusion matrix in the appropriate cv# folder/directory

    with open(os.path.join(os.path.dirname(training_matrix_pickle), 'trained_multinomialNB_classifier.pickle'), 'wb') as handle:
        pickle.dump(mnnb_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(os.path.dirname(training_matrix_pickle), 'confusion_matrix.pickle'), 'wb') as handle:
        pickle.dump(confusion_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_average_confusion_matrix(cv_directory, number_of_cvs=10):
    assert os.path.isdir(cv_directory)

    confusion_matrices_list = []
    for cv in range(number_of_cvs):
        with open(os.path.join(cv_directory, f"cv{cv}", "confusion_matrix.pickle"), 'rb') as cm_handle:
            confusion_matrices_list.append( pickle.load( cm_handle ) )

    # making sure all confusion matrices have the same shape
    for index in range(1,number_of_cvs):
        assert confusion_matrices_list[index].shape == confusion_matrices_list[index-1].shape
    
    average_confusion_matrix = np.zeros((confusion_matrices_list[0].shape))

    for matrix in confusion_matrices_list:
        for row_index, row in enumerate(matrix):
            for col_index, entry in enumerate(row):
                average_confusion_matrix[row_index, col_index] += float( entry / ( number_of_cvs ) )*100

    print(average_confusion_matrix)

    # an (int [0-19]) : (string topic) dictionary
    class_to_topic_mapping, _ = classify_topics(os.path.join("C:\\Users\\Samy\\Dropbox\\Samy_Dropbox\\MSc\\fall-2018-courses\\COMP-550\\project\\data_pickles", 'twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'))
    topics = [ class_to_topic_mapping[ i ] for i in range( len( confusion_matrices_list[0] ) ) ]

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
    matplotlib.rcParams.update({'font.size': 6})
    for col_index in range(len(average_confusion_matrix)):
        for row_index in range(len(average_confusion_matrix)):
            text = ax.text(row_index, col_index, 
                            '{0:.1f}'.format(average_confusion_matrix[col_index, row_index]), # 2 decimal places
                            ha="center", va="center", color="grey"
                        )

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
    pickle_path = '\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'data_pickles'))
    assert os.path.isdir(pickle_path)
    assert os.path.isfile(os.path.join(pickle_path, 'pickle_legend.txt'))
    for k in [0,1,2,3,4,5,6,7,8,9]:
        # To split the training and testing sets without the top 500 most frequent words

        print(f"iteration #{k}")
        #setup(pickle_path, iteration=k)
        
        training_testing_and_results_pickles_path = '\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'training_testing_and_results_pickles', '20_newsgroups', 'cv{}'.format(str(k))))
        #print(f"\n\n iteration #{k}, path_to_pickles: {training_testing_and_results_pickles_path} \n\n ")

        
        run_multinomialNB(
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_training_matrix_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_training_matrix_labels_vector_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_testing_matrix_cv#{}.pickle'.format(str(k))),
            os.path.join(training_testing_and_results_pickles_path, 'twenty_newsgroups_testing_matrix_labels_vector_cv#{}.pickle'.format(str(k)))
        )
        
    #commented-out just to generate the top-500-less matrices
    path_to_cv_directory = '\\\\?\\'+os.path.abspath(os.path.join(os.getcwd(), 'training_testing_and_results_pickles', '20_newsgroups'))
    generate_average_confusion_matrix(path_to_cv_directory)
    
    