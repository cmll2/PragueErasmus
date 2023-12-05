#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

def diacritize_word(input_word, vectorizer, clf):
    input_vector = vectorizer.transform([input_word])
    diacritized_word = clf.predict(input_vector)
    return diacritized_word[0]

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"  

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

def main(args: argparse.Namespace) -> Optional[str]:
    import re
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        #remove punctuation
        train_data = train.data.split()
        train_target = train.target.split()
        #keep only unique words so we don't overfit and their count
        unique_train_indices = np.unique(train_data, return_index=True)[1]
        unique_train_data = [train_data[index] for index in sorted(unique_train_indices)]
        unique_target_indices = np.unique(train_target, return_index=True)[1]
        unique_train_target = [train_target[index] for index in sorted(unique_target_indices)]
        #create vectorizer
        #for each word in train_data, create a key in dict in which the value is the most occurences of the word in train_target of this word diacritized
        def remove_dia(word):
            word = word.translate(Dataset.DIA_TO_NODIA)
            return word
        
        train_target = []
        for word in unique_train_data:
            #take all diacriticized versions of the word
            #print(word, remove_dia(word))
            possible_targets = [unique_train_target[index] for index in range(len(unique_train_target)) if word == remove_dia(unique_train_target[index])]
            #print(possible_targets)
            #count the occurences of each diacriticized version in the target data
            occurences = train_target.count(possible_targets)
            #take the most occuring diacriticized version
            most_occuring = possible_targets[np.argmax(occurences)]
            train_target.append(most_occuring)
        prediction_dict = dict(zip(train_data, train.target.split()))
        #keep only dictionnary items with diacritics
        prediction_dict = {key:value for key, value in prediction_dict.items() if key != value}
        prediction = [prediction_dict[word] if word in prediction_dict else word for word in train_data]
        #print(accuracy_score(prediction, train_target))
        dict_path = 'prediction_dict.pkl'
        with lzma.open(dict_path, "wb") as dict_file:
            pickle.dump(prediction_dict, dict_file)
            print(prediction[:15])
        prediction = ' '.join(prediction)
        return prediction

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open('prediction_dict.pkl', "rb") as dict_file:
            prediction_dict = pickle.load(dict_file)

        test_data = test.data.split()

        predicted = [prediction_dict[word] if word in prediction_dict else word for word in test_data]
        # convert predicted back to whole text
        predicted = ' '.join(predicted)
    return predicted

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
