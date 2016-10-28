import sys


class Logger():
    def __init__(self):
        pass

    def show_training_result(self, model):
        for params, mean_score, all_scores in model.grid_scores_:
            print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)
        print 'best parameter:', model.best_params_

    def show_exception(self, e):
        print "Exception:"
        print '  type     -> ', str(type(e))
        print '  args     -> ', str(e.args)
        print '  message  -> ', e.message
        print '  e        -> ', str(e)
        sys.exit(1)
