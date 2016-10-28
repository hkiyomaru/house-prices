import numpy as np
import pandas as pd


class Preprocessor():
    def __init__(self):
        pass

    def preprocess(self, csv):
        # fill NA/NaN
        csv = csv.fillna(0)
        # convert string to index
        vocab_dict = {}
        for column in csv.columns:
            current_column = csv[column]
            dtype = current_column.dtype
            if self.check_type(dtype): # string
                vocab_dict[column] = self.create_vocab(current_column)
        # mapping
        for key in vocab_dict.keys():
            csv[key] = csv[key].map(vocab_dict[key]).astype(int)
        return csv

    def create_vocab(self, column):
        vocab = {}
        vocab_n = 0
        for i in column:
            if vocab.has_key(i):
                continue
            else:
                vocab[i] = vocab_n
                vocab_n += 1
                continue
        return vocab

    def check_type(self, dtype):
        if dtype in ['int64', 'float64']:
            return False
        else:
            return True

if __name__ == '__main__':
    from CSVHandler import CSVHandler
    csv_handler = CSVHandler('../../data')
    preprocessor = Preprocessor()
    train = csv_handler.load_csv('train.csv')
    csv = preprocessor.preprocess(train)
    print csv
