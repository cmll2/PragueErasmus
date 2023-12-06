#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="data/nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        print('Before preprocessing: ', train.data[0])
        # TODO: Train a model on the given dataset and store it in `model`.
        train_data = train.data
        train_target = train.target

        #preprocess data
        import re
        for i in range(len(train_data)):
            train_data[i] = re.sub(r'[^\w\s]', '', train_data[i]) #remove punctuation
            train_data[i] = re.sub(r'\d+', '', train_data[i]) #remove numbers
            train_data[i] = train_data[i].lower() #lowercase
        #split data into train and test
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.1, random_state=42)

        #use vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 5), min_df=1, max_df=0.8, max_features=5000)
        vectorizer.fit(train_data)
        train_data = vectorizer.transform(train_data)
        test_data = vectorizer.transform(test_data)
        #Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
        model.fit(train_data, train_target)
        print('Gradient Boosting train accuracy: ', model.score(train_data, train_target))
        print('Gradient Boosting test accuracy: ', model.score(test_data, test_target))

        with lzma.open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        with lzma.open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        with lzma.open('model_gauss.pkl', 'rb') as model_file:
            model_gauss = pickle.load(model_file)

        with lzma.open('model_bern.pkl', 'rb') as model_file:
            model_bern = pickle.load(model_file)

        test_data = test.data
        
        predictions = []
        #soft voting
        for i in range(len(test_data)):
            nb = model.predict_proba(vectorizer.transform([test_data[i]]))[0]
            gauss = model_gauss.predict_proba(vectorizer.transform([test_data[i]]).toarray())[0]
            bern = model_bern.predict_proba(vectorizer.transform([test_data[i]]))[0]
            nb_index = np.argmax(nb)
            gauss_index = np.argmax(gauss)
            bern_index = np.argmax(bern)
            if nb[nb_index] > gauss[gauss_index] and nb[nb_index] > bern[bern_index]:
                predictions.append(nb_index)
            elif gauss[gauss_index] > nb[nb_index] and gauss[gauss_index] > bern[bern_index]:
                predictions.append(gauss_index)
            elif bern[bern_index] > nb[nb_index] and bern[bern_index] > gauss[gauss_index]:
                predictions.append(bern_index)
            else:
                predictions.append(nb_index)
        
        return np.asarray(predictions).flatten()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
