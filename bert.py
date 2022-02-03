#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : use pretrained bert to fill masks
'''


import pandas as pd
from transformers import pipeline


# debug prints
DBUG = True
# csv file data
CSV_FILES = ['A', 'B', 'C', 'D', 'E']
CSV_FOLDER = './csv/'
# load unmasking pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')


def debug(string, obj=None):
    """
    debug print that can be disabled w macro
    """
    if DBUG:
        print("[DEBUG] ")
        print(string)
        if obj is not None:
            print("\n")
            print(obj)


def load_sentences():
    '''
    load the masked sentences from the csv files
    '''
    all_sentences = []
    for csv_file in CSV_FILES:
        data = pd.read_csv(CSV_FOLDER+csv_file+'.csv')
        # get the last thing in the iterator to get sentences
        for item in data.items():
            pass
        sentences = item[1]
        all_sentences.extend(sentences)
    return all_sentences


def unmask_sentences(sentences):
    """
    unmask a set of sentences and return them as a list
    """
    unmasked_sentences = []
    # for sentence in sentences:
    #     result = unmasker("Hello, I'm a [MASK] model")
    for sentence in sentences:
        sentence += " ."
        results = unmasker(sentence)
        debug("predictions--")
        debug("original : " + sentence)
        for result in results:
            debug("prediction : " + result['sequence'])
            unmasked_sentences.extend(result['sequence'])
    return unmask_sentences


def unmask_sentence(sentence):
    """
    unmask and return a single sentences predictions
    """
    unmasked_sentences = []
    results = unmasker(sentence)
    debug("predictions--")
    debug("original : " + sentence + " .")
    for result in results:
        debug("prediction : " + result['sequence'])
        unmasked_sentences.extend(result['sequence'])
    return unmasked_sentences


def save_to_csv(sentences):
    """
    take a list of sentences and convert to csv
    """
    dictionary = {'sequence': sentences}
    dataframe = pd.DataFrame(dictionary)
    dataframe.to_csv(CSV_FOLDER + 'unmasked.csv/')


def main():
    """
    process all sentences for sentiment classification
    """
    save_to_csv(unmask_sentences(load_sentences()))
    # unmask_sentence("the disabled person has [MASK] .")


if __name__ == "__main__":
    main()
