import os, pickle
import pandas as pd 
import numpy as np 
from random import sample
from scipy.sparse import dok_matrix # sparse matrix data structure

def load_20newsgroups_lexicon(pickle_file_path='!20_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle'):
    '''
    Loads a pickled dictionary of corpus-wide (token:corpus-wide frequency) (key:value) pairs.
    Returns the list of lexicon tokens (w/o their frequency) sorted from most->least common

    Parameters:
        pickle_file_path: path to the pickled dictionary of corpus-wide (token:corpus-wide frequency)
            !20_newsgroups_corpus_wide_tokens_and_frequencies_(lexicon).pickle 
    
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

def classify_topics(pickle_file_path='!20_newsgroups_topic_to_list_of_topical_files.pickle'):
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
        '!20_newsgroups_topic_to_list_of_topical_files.pickle'
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
    for class_int, (topic, list_of_files_of_this_topic) in enumerate(list(topic_to_topical_document_IDs_dict.items())):
        #input(f"{topic}, {list_of_files_of_this_topic}")
        class_to_topic_mapping[class_int] = topic
        class_to_documents_dictionary[class_int] = list_of_files_of_this_topic

    return class_to_topic_mapping, class_to_documents_dictionary

def load_all_files_token_frequency_dictionaries(pickle_file_path='!20_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'):
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
        '!20_newsgroups_dict_of_dicts_docID_token_freq_dicts.pickle'
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

def main():

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

# 2. making the sparse training matrix

    training_sparse_matrix = dok_matrix(
        ( 
            sum( [ len(list_of_training_documents_for_this_class) for list_of_training_documents_for_this_class in training_set_document_IDs.values() ] ) ,
            len(lexicon)
        ), 
        dtype=np.float32
    )

# 3. populating the sparse training matrix
    for class_integer, class_document_list in training_set_document_IDs.items(): # could include a column for the class integer
        print(class_integer)
        for document_row, document in enumerate(class_document_list):
            for token, freq in docID_token_frequency_dictionaries[document].items():
                training_sparse_matrix[document_row, lexicon.index(token)] = freq
        
    print("done")
if __name__ == '__main__':
    main()