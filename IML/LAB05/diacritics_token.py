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
parser.add_argument("--model_path", default="diacritization_mlp.model", type=str, help="Model path")

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

LETTERS_SET = set(Dataset.LETTERS_NODIA + Dataset.LETTERS_DIA)

def accuracy(gold: str, system: str) -> float:
    assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"

    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), \
        "The gold and system outputs must have the same number of words: {} vs {}.".format(len(gold), len(system))

    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        correct += gold_token == system_token

    return correct / words

def create_target(target_text):
    #target_text = re.sub(r'[^\w\s]', '', target_text)
    target_text = [list(letter.lower()) for letter in target_text if letter.lower() in LETTERS_SET]
    return np.asarray(target_text)

def create_train_data(train_text):
    #train_text = re.sub(r'[^\w\s]', '', train_text)
    # 2-neighboors, so first should be '  xxx' and last 'xxx  '
    train_text = '    ' + train_text + '    '
    train_text = [list(train_text[i-4:i+5].lower())  for i in range(4, len(train_text)-4) if train_text[i].lower() in LETTERS_SET]
    return np.asarray(train_text)

def compute_accuracy(predicted, target):
    predicted = predicted.split()
    target = target.split()
    assert len(predicted) == len(target), "predicted and target have different length, {} != {}".format(len(predicted), len(target))
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == target[i]:
            correct += 1
    return correct / len(predicted)

def labelize_data(data): #function to label the characters for the model to learn
    train_data = create_train_data(data.data)
    train_target = create_target(data.target)
    print(train_data[0], train_data[-1], train_target[0], train_target[-1])
    #print(train_data[:10], train_target[:10])
    from sklearn.preprocessing import OneHotEncoder
    target_oh = OneHotEncoder()
    target_classes = set(LETTERS_SET)
    print('Classes :', target_classes)
    target_oh.fit(np.asarray(list(target_classes)).reshape(-1,1))
    train_classes = set(data.data).union('abcdedghijklmnopqrstuvwxyz1234567890,;:!?.- ')
    train_oh = OneHotEncoder()
    train_oh.fit(np.asarray(list(train_classes)).reshape(-1,1))
    print('Encoding train target ...')
    encoded_train_target = []
    encoded_train_target = target_oh.transform(train_target.reshape(-1,1)).toarray()
    encoded_train_target = np.asarray(encoded_train_target)
    encoded_train_target = encoded_train_target.reshape(encoded_train_target.shape[0], encoded_train_target.shape[1])
    print('Encoding train data ...')
    encoded_train_data = []
    for i in range(len(train_data)):
        encoded_train_data.append(train_oh.transform(np.asarray(train_data[i]).reshape(-1,1)).toarray())
    encoded_train_data = np.asarray(encoded_train_data)
    encoded_train_data = encoded_train_data.reshape(encoded_train_data.shape[0], encoded_train_data.shape[2]*encoded_train_data.shape[1])
    return encoded_train_data, train_target, train_oh, target_oh

def predict_text(text, oh_encoder, model, oh_decoder):
    print('Prediction process started.')
    text_to_predict = '    ' + text + '   '
    predicted = []
    for i in range(len(text_to_predict)):
        if text_to_predict[i].lower() in LETTERS_SET:
            #print(text_to_predict[i-4:i+5])
            label = oh_encoder.transform(np.asarray(list(text_to_predict[i-4:i+5].lower())).reshape(-1,1)).toarray()
            label = label.reshape(label.shape[1]*label.shape[0])
            prediction = model.predict(label.reshape(1,-1))
            #prediction  = oh_decoder.inverse_transform(prediction[0].reshape(1,-1))
            #check if text_to_predict[i] is uppercase
            if text_to_predict[i].isupper():
                predicted.append(prediction[0].upper())
            else:
                predicted.append(prediction[0])
        else:
            predicted.append(text_to_predict[i])
    return ''.join(predicted)
    
def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        print('Preparing the data...')
        train = Dataset()
        train_data, train_target, oh_encoder, oh_decoder = labelize_data(train)
        print('Training the model...')
        #sgd classifier grid_search
        model = sklearn.linear_model.SGDClassifier(loss='hinge', max_iter=args.epochs*20, tol=1e-3, random_state=args.seed)
        model.fit(train_data, train_target.ravel()) #ravel to avoid warning             
        print('Saving the model...')
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        with lzma.open('oh_encoder.pkl', "wb") as oh_encoder_file:
            pickle.dump(oh_encoder, oh_encoder_file)
        with lzma.open('oh_decoder.pkl', "wb") as oh_decoder_file:
            pickle.dump(oh_decoder, oh_decoder_file)
        # with lzma.open('oh_encoder.pkl', "rb") as oh_encoder_file:
        #     oh = pickle.load(oh_encoder_file)
        # with lzma.open(args.model_path, "rb") as model_file:
        #     model = pickle.load(model_file)
        predicted = predict_text(train.data, oh_encoder, model, oh_decoder)
        print(predicted[:100])
        print('Accuracy on train set :', accuracy(predicted, train.target))
    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        with lzma.open('oh_encoder.pkl', "rb") as oh_encoder_file:
            oh = pickle.load(oh_encoder_file)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with lzma.open('oh_decoder.pkl', "rb") as oh_decoder_file:
            oh_decoder = pickle.load(oh_decoder_file)
        predicted = predict_text(test.data, oh, model , oh_decoder)
    return predicted

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
