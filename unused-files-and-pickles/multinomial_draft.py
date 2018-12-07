import os, pickle
import pandas as pd 
import numpy as np 
from random import sample
from scipy.sparse import dok_matrix # sparse matrix data structure
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from collections import Counter


def load_20newsgroups_lexicon(pickle_file_path='twenty_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle'):
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

    lexicon = [ token for (token,frequency) in token_and_frequency_tuple_list ]
    return lexicon

def load_industry_sector_lexicon(pickle_file_path='industry_sector_tokens_and_frequencies_across_dataset_list_of_tuples_(lexicon).pickle'):
    '''
    Loads a pickled list of corpus-wide (token:corpus-wide frequency) tuples.
    Returns the list of lexicon tokens (w/o their frequency) sorted from most->least common

    Parameters:
        pickle_file_path: path to the pickled dictionary of corpus-wide (token:corpus-wide frequency)
            industry_sector_tokens_and_frequencies_across_dataset_list_of_tuples_(lexicon).pickle
    
    Returns:
        lexicon: the list of lexicon tokens sorted from most->least frequent in the corpus.
    '''
    # obtaining the lexicon (tokens) and sorting it by corpus-wide frequency    
    with open(pickle_file_path,'rb') as infile:
        corpus_wide_token_frequency_tuples_list = pickle.load(infile)
    
    # sorts the tuples list by frequency most->least
    corpus_wide_token_frequency_tuples_list.sort(key=lambda x:x[1], reverse=True)
    

    # counting the number of tokens which only occur once in the entire corpus
    '''unique_count = 0
    for (token,frequency) in token_and_frequency_tuple_list[::-1]:
        if frequency > 3:
            break
        else:
            unique_count += 1
    print(unique_count)'''

    lexicon = [ token for (token,frequency) in corpus_wide_token_frequency_tuples_list ]
    return lexicon

def classify_topics(pickle_file_path='twenty_newsgroup_dict_of_dicts_of_topic_and_topical_file_name_as_keys_and_file_valid_lines_as_values.pickle'):
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
        #input(f"{topic}, {filename_and_valid_lines_dictionary}")
        class_to_topic_mapping[class_int] = topic
        class_to_documents_dictionary[class_int] = list(filename_and_valid_lines_dictionary.keys())

    return class_to_topic_mapping, class_to_documents_dictionary

def load_all_files_token_frequency_dictionaries(pickle_file_path='twenty_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'):
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

def setup():

# obtaining the lexicon (tokens) and sorting it by corpus-wide frequency    
    lexicon = load_20newsgroups_lexicon()

# mapping the 20 topics to integer classes [0-19]
    class_to_topic_mapping, class_to_documents_dictionary = classify_topics()

# loading the token-frequency dictionaries for all files
    docID_token_frequency_dictionaries = load_all_files_token_frequency_dictionaries()
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

# for one iteration of the k-fold cross validation process:

# 1. splitting data into training/testing
# assumes an 80/20 split
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

    training_labels_vector =[0]*number_of_training_documents
    testing_labels_vector = [0]*number_of_testing_documents
    print(f"{number_of_training_documents} training documents, size of label array = {len(training_labels_vector)}")
    print(f"{number_of_testing_documents} testing documents, size of label array = {len(testing_labels_vector)}")
    
    for class_integer, class_document_list in training_set_document_IDs.items(): # could include a column for the class integer label
        
        print(f"Class: {class_integer}: {class_to_topic_mapping[class_integer]}")

        for document_row, document in enumerate(class_document_list):
            #print(f"examining document {document}")
            training_labels_vector[document_row] = class_integer
            # uncomment if you just want to get the labels 
            break 
            for token, freq in docID_token_frequency_dictionaries[document].items():
                training_sparse_matrix[document_row, lexicon.index(token)] = freq
                
                # debugging code, can be ignored
                
                #print(f"token: {token}, freq: {freq}")
                #input(training_sparse_matrix[document_row, lexicon.index(token)])
            
            #print('\n'.join(list(map(str, list(docID_token_frequency_dictionaries['53513'].items())))))
            #print('\n\n\n')
            #if document_row == 0:
                #for col_index in range(len(lexicon)):
                    #if training_sparse_matrix[0,col_index] > 0:
                        #print(f"('{lexicon[col_index]}', {training_sparse_matrix[0,col_index]})")
                #break
            #break
        #break
        
    with open('twenty_newsgroups_training_matrix.pickle','wb') as handle:
        pickle.dump(training_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('twenty_newsgroups_training_matrix_labels_vector.pickle','wb') as handle:
        pickle.dump(training_labels_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("done populating the training matrix")

# 4. populating the sparse testing matrix
    
    for class_integer, class_document_list in testing_set_document_IDs.items():
        print(f"Class: {class_integer}: {class_to_topic_mapping[class_integer]}")

        for document_row, document in enumerate(class_document_list):
            testing_labels_vector[document_row] = class_integer
            # uncomment if you just want to get the labels 
            break
            for token, freq in docID_token_frequency_dictionaries[document].items():
                testing_sparse_matrix[document_row, lexicon.index(token)] = freq
        
    with open('twenty_newsgroups_testing_matrix.pickle','wb') as handle:
        pickle.dump(testing_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('twenty_newsgroups_testing_matrix_labels_vector.pickle','wb') as handle:
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

    testlabels = Counter(testing_labels)
    print(testlabels)
    trainlabels = Counter(training_labels)
    print(trainlabels)
    # don't forget to permute the training matrix and labels together!

    try:
        assert training_matrix.shape[0] == len(training_labels)
    except AssertionError:
        print(f"length of training matrix = {len(training_matrix)}, length of training labels row matrix = {len(training_labels)}")

    mnnb_classifier = MultinomialNB(alpha=0.01,fit_prior=False) # to match the paper's multinomial classifier's parameters
    mnnb_classifier.fit(training_matrix, training_labels)
    print("done training the classifier")

    predictions = mnnb_classifier.predict(testing_matrix)
    input(predictions.shape)

    
    #confusion_matrix = metrics.confusion_matrix(testing_labels, predictions, labels=list(testing_labels))
    
    #print(confusion_matrix)


if __name__ == '__main__':
    setup()
    run_multinomialNB('twenty_newsgroups_training_matrix.pickle','twenty_newsgroups_training_matrix_labels_vector.pickle','twenty_newsgroups_testing_matrix.pickle','twenty_newsgroups_testing_matrix_labels_vector.pickle')
    
    
    