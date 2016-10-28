import pandas as pd
import os

class CSVHandler():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_csv(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath, index_col='Id')

if __name__ == '__main__':
    csv_handler = CSVHandler('../../data')
    train = csv_handler.load_csv('train.csv')
    print train.query('index == 1')
