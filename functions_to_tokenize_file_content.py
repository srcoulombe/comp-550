import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')

def has_some_alphanumeric_characters(line):
    if re.search('[a-zA-Z]', line):
        return True
    else:
        return False

def tokenize(list_of_lines, verbose=False):
    tokenized_list = []
    for line in list_of_lines:
        tokens = [word for word in nltk.word_tokenize(line) if has_some_alphanumeric_characters(word)]
        if verbose:
            print('The line "{}" becomes:\n{}\n\n'.format(str(line), ', '.join(tokens)))
        for tok in tokens:
            tokenized_list.append(tok)
    return tokenized_list