import sys
import os

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

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
    train_target = train["SalePrice"].values
    train, test = preprocessor.preprocess(train, test)

    # print "save test ids"
    test_ids = test.index

    # print "extract target column and feature column for both data"
    train_feature = train.values
    test_feature = test.values

    # print "train"
    tuned_parameters = [{'C': [1, 10, 100, 1000], 'epsilon': [1e-3, 1e-4, 1e-5]}]
    reg = GridSearchCV(
        SVR(),
        tuned_parameters,
        cv=5
    )
    reg.fit(train_feature, train_target)

    for params, mean_score, all_scores in reg.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

    print 'best parameter:', reg.best_params_

    # print "test"
    predict = reg.predict(test_feature).astype(float)

    # save
    output = zip(test_ids, predict)
    csv_handler.save_csv(output)

    # success
    return 0

if __name__ == '__main__':
    sys.exit(main())
