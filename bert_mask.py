#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : use pretrained bert to fill masks on templates from csv files
'''


import pandas as pd
from transformers import pipeline
from tqdm import tqdm


# debug toggle
DBUG = False
# csv file data
CSV_FILES = ['A', 'B', 'C', 'D', 'E']
CSV_FOLDER = './csv/'
# load unmasking pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')


def debug(string, obj=None):
    """
    debug print that can be disabled w constant
    """
    if DBUG:
        print("[DEBUG] ")
        print(string)
        if obj is not None:
            print("\n")
            print(obj)


def load_csv_files():
    '''
    load the masked sentences from the csv files
    '''
    file_data = []
    for csv_file in CSV_FILES:
        data = pd.read_csv(CSV_FOLDER+csv_file+'.csv')
        # get the last thing in the iterator to get sentences
        for item in data.items():
            pass
        wrapper = []
        wrapper.extend(item[1])
        file_data.append(wrapper)
        print("loaded " + csv_file + ".csv")
    return file_data


def unmask_csv_files(csv_data):
    """
    unmask all csv files and return as 2d list
    """
    unmasked_data = []
    for csv in csv_data:
        print("unmasking csv...")
        unmasked_data.append(unmask_sentences(csv))
    return unmasked_data


def unmask_sentences(sentences):
    """
    unmask a set of sentences and return them as a list
    """
    unmasked_sentences = []
    for sentence in tqdm(sentences):
        sentence += " ."  # messy fix to prevent predicting puncuation :(
        results = unmasker(sentence)
        debug("predictions--")
        debug("original : " + sentence)
        for result in results:
            debug("prediction : " + result['sequence'])
            unmasked_sentences.append(result['sequence'])
    return unmasked_sentences


def unmask_sentence(sentence):
    """
    unmask and return a single sentences predictions
    """
    unmasked_sentences = []
    sentence += " ."  # messy fix to prevent predicting puncuation :(
    results = unmasker(sentence)
    debug("predictions--")
    debug("original : " + sentence + " .")
    for result in results:
        debug("prediction : " + result['sequence'])
        unmasked_sentences.extend(result['sequence'])
    return unmasked_sentences


def save_to_csv(csv_structure):
    """
    take a list of sentences and convert to csv
    """
    for i, sentences in enumerate(csv_structure):
        dictionary = {'sequence': sentences}
        dataframe = pd.DataFrame(dictionary)
        dataframe.to_csv(CSV_FOLDER + CSV_FILES[i] + 'unmasked.csv')


def generate_data():
    """
    process all sentences for sentiment classification
    """
    print("loading csv files...")
    data = load_csv_files()
    print("csv files loaded.\nmaking predictions...")
    predictions = unmask_csv_files(data)
    print("made predictions.\nsaving to csv...")
    save_to_csv(predictions)
    print("saved to csv.")
    return predictions


def main():
    """
    main, generate data and do junk with it
    """
    data = generate_data()
    print(data)
    # unmask_sentence("the disabled person has [MASK] .")


if __name__ == "__main__":
    main()
