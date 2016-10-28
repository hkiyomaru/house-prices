import numpy as np
import pandas as pd


class Preprocessor():
    def __init__(self):
        pass

    def preprocess(self, train, test):
        # fill NA/NaN
        train = train.fillna(0)
        test = test.fillna(0)
        # convert string to index
        vocab_dict = {}
        for column in test.columns:
            dtype = test[column].dtype
            if self.check_type(dtype): # string
                vocab_dict[column] = self.create_vocab(train[column], test[column])
        # mapping
        for key in vocab_dict.keys():
            train[key] = train[key].map(vocab_dict[key]).astype(int)
            test[key] = test[key].map(vocab_dict[key]).astype(int)
        return train, test

    def create_vocab(self, train_column, test_column):
        vocab = {}
        vocab_n = 0
        for i in train_column:
            if vocab.has_key(i):
                continue
            else:
                vocab[i] = vocab_n
                vocab_n += 1
                continue
        for i in test_column:
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
    test = csv_handler.load_csv('test.csv')
    train, test = preprocessor.preprocess(train, test)
    print train, test
