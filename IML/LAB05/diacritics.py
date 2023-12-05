#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose
import numpy as np

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self, name="fiction-train.txt", url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print(f"Downloading dataset {name}...")
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

keys = 'abcdefghijklmnopqrstuvwxyz.?!-"'
vowels = 'aeiouy'
consonants = 'bcdfghjklmnpqrstvwxz'
tokens_dictionary = dict(zip(keys, range(len(keys))))
diacritics_pairs = {'a': ['a', '', 'á', ''], 'c': ['c', 'č', '', ''], 'd': ['d', 'ď', '', ''],
                    'e': ['e', 'ě', 'é', ''], 'i': ['i', '', 'í', ''], 'n': ['n', 'ň', '', ''],
                    'o': ['o', '', 'ó', ''], 'r': ['r', 'ř', '', ''], 's': ['s', 'š', '', ''],
                    't': ['t', 'ť', '', ''], 'u': ['u', '', 'ú', 'ů'], 'y': ['y', '', 'ý', ''],
                    'z': ['z', 'ž', '', '']}

def tokenize_sentences(data):
    tokenized_data = []
    meta_data = []
    for sentence in data:
        sentence_metadata = [len(sentence), len(sentence.split(' '))]
        tokenized_sentence = []
        words_meta_data = []
        for word in sentence.lower().split(' '):
            word_meta_data = [len(word), count_vowels(word), count_consonants(word)]
            tokenized_word = [tokens_dictionary.get(letter, -1) for letter in word]
            words_meta_data.append(word_meta_data)
            tokenized_sentence.append(tokenized_word)

        sentence_metadata.append(words_meta_data)
        meta_data.append(sentence_metadata)
        tokenized_data.append(tokenized_sentence)
    return tokenized_data, meta_data

def count_vowels(string):
    return len([char for char in string if char in vowels])

def count_consonants(string):
    return len([char for char in string if char in consonants])

def create_features(tokenized_data, metadata):
    features = []
    feature_indexes = 0
    for sentence_i, sentence in enumerate(tokenized_data):
        for word_i, word in enumerate(sentence):
            for letter_index, letter_token in enumerate(word):
                if letter_token != -1 and keys[letter_token] in diacritics_pairs:
                    feature_indexes += 1
                    row = [letter_index, metadata[sentence_i][:2], metadata[sentence_i][1][word_i],
                           metadata[sentence_i][0] if word_i-1 >= 0 else [0, 0],
                           metadata[sentence_i][0] if word_i+1 < len(metadata[sentence_i][1]) else [0, 0],
                           letter_token]
                    for i in range(1, 8):
                        row.append(word[letter_index-i] if letter_index-i >= 0 else -1)
                    for i in range(1, 8):
                        row.append(word[letter_index+i] if letter_index+i < len(word) else -1)
                    features.append(row)
    return np.array(features), feature_indexes

def create_targets(nondia, dia):
    results = []
    for sentence_i, sentence in enumerate(nondia):
        for letter_i, letter in enumerate(sentence):
            if letter.lower() in diacritics_pairs:
                result = [i for i, char in enumerate(diacritics_pairs[letter.lower()]) if char == dia[sentence_i][letter_i].lower()]
                if not result:
                    raise Exception('Letter not found')
                results.append(result[0])
    return results

def accuracy(gold, system):
    assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"
    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), "The gold and system outputs must have the same number of words"
    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        correct += gold_token == system_token
    return correct / words

def main(args):
    if args.predict is None:
        np.random.seed(args.seed)
        dataset = Dataset()
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            dataset.data.split('\n'), dataset.target.split('\n'), test_size=0.001, random_state=args.seed)
        tokenized_data, metadata = tokenize_sentences(train_data)
        features, feature_indexes = create_features(tokenized_data, metadata)
        targets = create_targets(train_data, train_target)
        test_tokenized_data, test_metadata = tokenize_sentences(test_data)
        test_features, indexes = create_features(test_tokenized_data, test_metadata)
        test_targets = create_targets(test_data, test_target)
        parameters = {}  # {'estimator__activation': ['relu', 'logistic']}
        pipeline = sklearn.pipeline.Pipeline([
            ('transformer', sklearn.compose.ColumnTransformer([('encoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), [*range(10,25)])], remainder='passthrough')),
            ("estimator", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(700, 400), verbose=True, random_state=1)),
        ])
        model = pipeline.fit(features, targets)
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        test_predictions = model.predict(test_features)
        test_accuracy = sklearn.metrics.accuracy_score(test_predictions, test_targets)
        print(test_accuracy)
    else:
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        index = 0
        test_sentences = test.data.split('\n')
        evaluate_data = []
        result = ''
        while index < len(test_sentences):
            evaluate_data.append(test_sentences[index])
            if index % 60 == 59 or index == len(test_sentences)-1:
                evaluate_raw = '\n'.join(evaluate_data)
                if index != 59:
                    result += '\n'
                tokenized_data, metadata = tokenize_sentences(evaluate_data)
                features, feature_indexes = create_features(tokenized_data, metadata)
                count = 0
                real_count = 0
                predictions = model.predict_proba(features)
                for letter in evaluate_raw.lower():
                    if letter in diacritics_pairs.keys():
                        prediction = predictions[count]
                        sorted_indexes = np.argsort(-prediction)
                        i = 0
                        while diacritics_pairs[letter][sorted_indexes[i]] == '':
                            i += 1
                        final = diacritics_pairs[letter][sorted_indexes[i]]
                        if letter != evaluate_raw[real_count]:
                            final = final.upper()
                        result += final
                        count += 1
                    else:
                        result += evaluate_raw[real_count]
                    real_count += 1
                evaluate_data = []
            index += 1
        return result

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)