# Import block
import os
import pickle
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt') # link to documentation on punkt tokenizers: c

# first part
def checkUtfEncoding():
    with open(os.path.abspath(os.path.join(os.getcwd(), "20_newsgroups", "alt.atheism", "49960")), errors='ignore') as atheism_sample_file_2:
        print(''.join(atheism_sample_file_2.readlines()))
    return

def checkSubjects():
    root_dir = os.path.abspath( os.path.join( os.getcwd(), '20_newsgroups') )

    file_count = 0
    missing_subject_line_count = 0
    subdirectory_list = ['alt.atheism']#, 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    for subdir in subdirectory_list:
        files = [ os.path.join(root_dir, subdir, f) for f in os.listdir(os.path.join(root_dir, subdir)) ]
        for current_file in files:
            with open(current_file,'r', errors='ignore') as input_file: # on MacOS: added error case handling
                contents = input_file.readlines()
            found_subject_line = False
            for line in contents:
                if "subject:" in line.lower():
                    found_subject_line = True
                    break
            if not found_subject_line:
                print(f"{current_file} didn't have a subject line.")
                missing_subject_line_count += 1
            file_count += 1

    print(f"examined a total of {file_count} files, {missing_subject_line_count} of which didn't have a subject line.")
    return


def populateSuperDictionary():
    root_dir = os.path.abspath(os.path.join(os.getcwd(), '20_newsgroups'))
    file_count = 0
    subdirectory_list = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    super_dictionary = {}

    for subdir in subdirectory_list:
        super_dictionary[subdir] = dict([]) # create empty dictionary
        files = [ os.path.join(root_dir, subdir, f) for f in os.listdir(os.path.join(root_dir, subdir)) ]
        for current_file in files:
            lines_with_valid_content = []
            with open(current_file,'r', errors='ignore') as input_file:
                contents = input_file.readlines()

            for i, line in enumerate(contents):
                if "lines: " in line.lower():
                    lines_with_valid_content.append(line.rstrip())
                    break
            i += 1
            while i < len(contents):
                lines_with_valid_content.append(contents[i].rstrip())
                i += 1

            subdirectory_dictionary = super_dictionary[subdir]
            subdirectory_dictionary[os.path.basename(current_file)] = lines_with_valid_content

            file_count += 1

    print(f"examined a total of {file_count} files.")
    return super_dictionary

def checkSuperDictionary( superDictionary ):
    alt_atheism_dictionary = superDictionary["alt.atheism"]
    print('\n'.join(alt_atheism_dictionary["49960"]))
    return

def checkHtmlIndustrySector():
    root_dir = os.path.abspath(os.path.join(os.getcwd(), 'sector'))
    openable_file_count = 0
    file_count = 0
    body_tag_count = 0
    strangely_encoded_files = []
    for dirname, dirnames, filenames in os.walk(root_dir):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            #print(os.path.join(dirname, subdirname))
            pass

        # print path to all filenames.
        file_count += len(filenames)
        for filename in filenames:
            with open(os.path.join(dirname, filename),'r', errors='ignore') as infile:
                contents = infile.readlines()
                for line in contents:
                    if '<body' in line.lower():
                        body_tag_count += 1
                        break
                openable_file_count += 1
                    
    print(f"examined {openable_file_count}/{file_count} and counted {body_tag_count} <body> tags")
    print('failed to open {} files:{}'.format(str(len(strangely_encoded_files)), '\n\n'.join(strangely_encoded_files)))
    return

def michin():
    return
