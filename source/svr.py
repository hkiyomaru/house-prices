import sys
import os
from sklearn.svm import SVR

from utils import CSVHandler
from utils import Preprocessor

"""
File paths
"""
data_dir = '../data/'
train_filename = 'train.csv'
test_filename = 'test.csv'

"""
Make directory to save result
"""
try:
    os.mkdir('./predict/')
except:
    pass


# main loop
def main():
    csv_handler = CSVHandler.CSVHandler(data_dir)
    preprocessor = Preprocessor.Preprocessor()

    # print "load train data and test data"
    try:
        train = csv_handler.load_csv(train_filename)
        test = csv_handler.load_csv(test_filename)
    except Exception as e:
        print "Exception:"
        print '  type     -> ', str(type(e))
        print '  args     -> ', str(e.args)
        print '  message  -> ', e.message
        print '  e        -> ', str(e)
        return 1

    # print "preprocess the both data"
    train = preprocessor.preprocess(train)
    test = preprocessor.preprocess(test)

    # print "save test ids"
    test_ids = test.index

    # print "extract target column and feature column for both data"
    train_target = train["SalePrice"].values
    train_feature = train.drop("SalePrice", axis=1).values
    test_feature = test.values

    # print "train"
    svr = SVR(C=10000, epsilon=0.1)
    svr.fit(train_feature, train_target)

    # print "test"
    predict = svr.predict(test_feature).astype(float)

    # save
    output = zip(test_ids, predict)
    csv_handler.save_csv(output)

    # success
    return 0

if __name__ == '__main__':
    sys.exit(main())
