import argparse
import pickle
import lzma
import numpy as np

import re

def create_train_data(train_text):
    train_text = train_text.replace('\n', ' ')
    train_text = re.sub(r'[^\w\s]', '', train_text)
    # 2-neighboors, so first should be '  xxx' and last 'xxx  ' to get all 3-grams
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

diacritics_dictionnary = diacritics_lower_dictionnary.copy()
diacritics_dictionnary.update(diacritics_upper_dictionnary)

def diacritize_letter_proba(input, vectorizer, clf, test_data):
    #print(input)
    proba_prediction = clf.predict_proba(input)
    #sort the probabilities
    sorted_proba = np.argsort(proba_prediction[0])
    #get the 3 most probable letters
    most_probable = [sorted_proba[-1], sorted_proba[-2], sorted_proba[-3]]
    #print(clf.classes_[most_probable[0]], clf.classes_[most_probable[1]], clf.classes_[most_probable[2]])
    #check in the dictionary if the letter is in the 3 most probable
    if diacritics_dictionnary.get(input[2]) is not None:
        for index in most_probable:
            if clf.classes_[index] in diacritics_dictionnary[input[2]]:
                return clf.classes_[index]
    return input[2]

def diacritize_letter(input, vectorizer, clf) -> str:

    prediction = clf.predict(vectorizer.transform([input]))
    #print(prediction, input[2])
    if diacritics_dictionnary.get(input[2]) is not None:
        if prediction in diacritics_dictionnary[input[2]]:
            return prediction.tolist()[0]
        else:
            return input[2]
    return input[2]

def predict_data(model, vectorizer, data):
    LETTERS_NODIA = "acdeeinorstuuyz"
    data = data.replace('\n', ' ')
    data = re.sub(r'[^\w\s]', '', data)
    data = '  ' + data + '  '
    predicted_text = []
    for i in range(2 ,len(data)-3):
        if data[i] in LETTERS_NODIA:
            char = model.predict(vectorizer.transform([data[i-2:i+3]])).tolist()[0]
            predicted_text.append(char) if char in diacritics_dictionnary[data[i]] else predicted_text.append(data[i])
        else:
            predicted_text.append(data[i])
    predicted_text = ''.join(predicted_text)
    return predicted_text

def predict_data_dictionnary(dictionary_model, data):
    data = data.replace('\n', ' ')
    data = re.sub(r'[^\w\s]', '', data)
    predicted_text = [''] * len(data)
    data = '  ' + data + '  '
    for i in range(2 ,len(data)-3):
        #print(i/len(data) * 100, '%')
        print(i)
        predicted_text.append(dictionary_model[data[i]][1].predict(dictionary_model[data[i]][0].transform([data[i-2:i+3]]))[0])
        print(predicted_text)
    predicted_text = ''.join(predicted_text)
    return predicted_text

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
    
def main():

    with lzma.open('dictionary_model.pkl', 'rb') as f:
        dictionary_model = pickle.load(f)


    test_data = 'Pomalu otocil hlavu a pres rameno zavolal do domu : \n Pomalu otocil hlavu a pres rameno zavolal do domu : \n az zer ert gfisdgfgsdfgjsdgfgsjf'
    # print("Test data:")
    # print(test_data)
    # test_data = create_train_data(test_data)
    # predicted_proba = ''
    # true_prediction = clf.predict(vectorizer.transform(test_data))
    # print("True prediction:")
    # print(''.join(true_prediction))
    # for i in range(test_data.shape[0]):
    #     #predicted_proba.append(diacritize_letter_proba(test_data_vectorized[i], vectorizer, clf, test_data[i]))
    #     predicted.append(diacritize_letter(test_data[i], vectorizer, clf))
    # print("Predicted proba:")
    # print(''.join(predicted_proba))
    # print("Predicted:")
    # print(''.join(predicted))

    # return ' '.join(predicted)
    predicted = predict_data_dictionnary(dictionary_model, test_data)
    return predicted

if __name__ == "__main__":
    pred = main()
    print(pred)