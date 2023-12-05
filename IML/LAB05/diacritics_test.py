#!/usr/bin/env python6
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
import re
import sklearn.neural_network

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"  

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2624/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()   
        self.data = self.target.translate(self.DIA_TO_NODIA)

def create_target(target_text):
    target_text = target_text.replace('\n', ' ')
    target_text = re.sub(r'[^\w\s]', '', target_text)
    target_text = [letter for letter in target_text]
    return np.asarray(target_text[0:-1])

def create_train_data(train_text):
    train_text = train_text.replace('\n', ' ')
    train_text = re.sub(r'[^\w\s]', '', train_text)
    # 2-neighboors, so first should be '  xxx' and last 'xxx  '
    train_text = '  ' + train_text + '  '
    train_text = [train_text[i-2:i+3] for i in range(2, len(train_text)-3)]
    return np.asarray(train_text)

def compute_accuracy(predicted, target):
    assert len(predicted) == len(target), "predicted and target have different length"
    for i in range(len(predicted)):
        if predicted[i] == target[i]:
            correct += 1
    return correct / len(predicted)

diacritics_lower_dictionnary = {
    'a': ['a','á'],
    'c': ['c', 'č'],
    'd': ['d','ď'],
    'e': ['e','ě', 'é'],
    'i': ['i','í'],
    'n': ['n','ň'],
    'o': ['o','ó'],
    'r': ['r','ř'],
    's': ['s','š'],
    't': ['t', 'ť'],
    'u': ['u''ú','ů'],
    'y': ['y','ý'],
    'z': ['z', 'ž'] 
}

diacritics_upper_dictionnary = {
    'A': ['A','Á'],
    'C': ['C', 'Č'],
    'D': ['D','Ď'],
    'E': ['E','Ě', 'É'],
    'I': ['I','Í'],
    'N': ['N','Ň'],
    'O': ['O','Ó'],
    'R': ['R','Ř'],
    'S': ['S','Š'],
    'T': ['T', 'Ť'],
    'U': ['U','Ú','Ů'],
    'Y': ['Y','Ý'],
    'Z': ['Z', 'Ž'] 
}

#combine both to have only one dictionnary
diacritics_dictionnary = diacritics_lower_dictionnary.copy()
diacritics_dictionnary.update(diacritics_upper_dictionnary)

def diacritize_letter(input, vectorizer, clf):
    prediction = clf.predict(vectorizer.transform([input]))
    if input[2] in diacritics_lower_dictionnary:
        if prediction in diacritics_lower_dictionnary[input[2]]:
            return prediction
        else:
            return input[2]
    if input[2] in diacritics_upper_dictionnary:
        if prediction in diacritics_upper_dictionnary[input[2]]:
            return prediction
        else:
            return input[2]
    return input[2]

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        print('Preparing the data...')
        train = Dataset()
        print(train.target[:100])
        #create target
        train_target = create_target(train.target)
        #create train data
        train_data = create_train_data(train.data)
        #print last 10 train data and target
        print(train_data[-10:])
        print(train_target[-10:])
        print('Vectorizing the data...')
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
        vectorizer.fit(train_data)
        X = vectorizer.transform(train_data)
        print(X.shape[0], X.shape[1])
        #train the model
        print('Training the model...')
        #model linear regression
        # model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,100), max_iter=100)
        model.fit(X, train_target)
        #save the model
        print('Saving the model...')
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        with lzma.open('vectorizer.pkl', "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        #predict the train set
        print('Predicting the train set...')
        true_predictions = model.predict(X)
        print(true_predictions[:100])
    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        test_data = create_train_data(test.data)
        with lzma.open('vectorizer.pkl', "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        predicted = []
        for i in range (test_data.shape[0]):
            predicted.append(diacritize_letter(test_data[i], vectorizer, model))
        predicted = ''.join(predicted)
    return predicted

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
