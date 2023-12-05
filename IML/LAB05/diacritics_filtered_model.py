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

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"  

def create_target(target_text):
    target_text = target_text.replace('\n', ' ')
    target_text = re.sub(r'[^\w\s]', '', target_text)
    target_text = [letter for letter in target_text if letter.lower() in LETTERS_DIA or letter.lower() in LETTERS_NODIA]
    return np.asarray(target_text)

def create_train_data(train_text):
    train_text = train_text.replace('\n', ' ')
    train_text = re.sub(r'[^\w\s]', '', train_text)
    # 2-neighboors, so first should be '  xxx' and last 'xxx  '
    train_text = '  ' + train_text + '  '
    train_text = [train_text[i-2:i+3] for i in range(2, len(train_text)-3) if train_text[i].lower() in LETTERS_NODIA]
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

def predict_data(model, vectorizer, data):
    LETTERS_NODIA = "acdeeinorstuuyz"
    data = data.replace('\n', ' ')
    data = re.sub(r'[^\w\s]', '', data)
    data = '  ' + data + '  '
    predicted_text = []
    for i in range(2 ,len(data)-3):
        print(i/len(data) * 100, '%')
        if data[i] in LETTERS_NODIA:
            char = model.predict(vectorizer.transform([data[i-2:i+3]])).tolist()[0]
            predicted_text.append(char) # if char in diacritics_dictionnary[data[i]] else predicted_text.append(data[i])
        else:
            predicted_text.append(data[i])
    predicted_text = ''.join(predicted_text)
    return predicted_text

ALPHABET_lower = 'abcdefghijklmnopqrstuvxyz'
ALPHABET_upper = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'

def create_keys(keys): #add alphabet letters as keys to a dict
    keys_dict = {}
    for char in keys:
        keys_dict[char] = []
    return keys_dict
    
class Model: #a model for each letter that doesn't change that has a predict method returning the letter
    def __init__(self, letter):
        self.letter = letter
    def predict(self, data):
        return self.letter
    
class Vectorizer: #a vectorizer for each letter that doesn't change that has a fit_transform method returning the letter
    def __init__(self, letter):
        self.letter = letter
    def transform(self, data):
        return self.letter

def create_dictionnary_model(all_chars, train_data, train_target, vectorizer, model):
    dictionary_model = create_keys(all_chars)
    dictionary_target = create_keys(all_chars)
    dictionary_data = create_keys(all_chars)
    print('Creating the dictionary dataset...')
    for i in range(len(train_data)):
        letter_to_predict = train_data[i][2]
        dictionary_target[letter_to_predict].append(train_target[i])
        dictionary_data[letter_to_predict].append(train_data[i])
    print(dictionary_data['a'][:10])
    print(dictionary_target['a'][:10])
    print(dictionary_data['e'][:10])
    print(dictionary_target['e'][:10])
    print(dictionary_data['i'][:10])
    print(dictionary_target['i'][:10])
    print('Vectorizing the train data...')
    for letter in dictionary_data:
        if len(dictionary_data[letter]) > 0:

            dictionary_model[letter].append(vectorizer.fit(dictionary_data[letter]))
            dictionary_data[letter] = dictionary_model[letter][0].transform(dictionary_data[letter])
        else:
            dictionary_model[letter].append(Vectorizer(letter))
    print('Training the models...')
    for letter in dictionary_model:
        print(letter, dictionary_target[letter][:10])
        if len(dictionary_target[letter]) > 0:
            dictionary_model[letter].append(model.fit(dictionary_data[letter], dictionary_target[letter]))
        else:
            dictionary_model[letter].append(Model(letter))
    return dictionary_model

def predict_data_dictionnary(dictionary_model, data):
    data = data.replace('\n', ' ')
    data = re.sub(r'[^\w\s]', '', data)
    data = '  ' + data + '  '
    predicted_text = []
    for i in range(2 ,len(data)-3):
        print(data[i]), print(dictionary_model[data[i]][1].predict(dictionary_model[data[i]][0].transform([data[i-2:i+3]]))[0])
        predicted_text.append(dictionary_model[data[i]][1].predict(dictionary_model[data[i]][0].transform([data[i-2:i+3]]))[0])
        #print(predicted_text)
    predicted_text = ''.join(predicted_text)
    return predicted_text


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        print('Preparing the data...')
        train = Dataset()
        print(train.target[:100])
        #create target
        all_chars = set(train.data).union(set('1234567890'))
        train_target = create_target(train.target)
        #create train data
        train_data = create_train_data(train.data)
        print(len(train_data), len(train_target))
        #print last 10 train data and target
        print(train_data[-10:])
        print(train_target[-10:])
        #create dictionary model
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
        dictionary_model = create_dictionnary_model(all_chars, train_data, train_target, vectorizer, model)
        #save the model
        print('Saving the model...')
        with lzma.open('dictionary_model.pkl', "wb") as dictionary_model_file:
            pickle.dump(dictionary_model, dictionary_model_file)
        #predict the train set
        print('Predicting the train set...')
        predicted = predict_data_dictionnary(dictionary_model, train.data) 
        print(predicted[:100])
        return predicted
    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open('dictionary_model.pkl', "rb") as dictionary_model_file:
            dictionary_model = pickle.load(dictionary_model_file)
        predicted = predict_data_dictionnary(dictionary_model, test.data)
    return predicted

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
