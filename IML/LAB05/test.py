import argparse
import pickle
import lzma
import numpy as np


def main():

    with lzma.open('diacritization.model', "rb") as model_file:
            model = pickle.load(model_file)

    with lzma.open('vectorizer.pkl', "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with lzma.open('label_encoder.pkl', "rb") as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    test_data = 'Pomalu otocil hlavu a pres rameno zavolal do domu :'
    test_data = test_data.split()
    test_data_matrix = vectorizer.transform(test_data)
    predicted_label = model.predict(test_data_matrix)
    predicted_diacritic = label_encoder.inverse_transform(predicted_label)

    return " ".join(predicted_diacritic)

if __name__ == "__main__":
    pred = main()
    print(pred)