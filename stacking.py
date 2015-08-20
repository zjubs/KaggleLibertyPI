from __future__ import division
import pandas as pd
import numpy as np
#import load_data
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import linear_model

#def logloss(attempt, actual, epsilon=1.0e-15):
#    """Logloss, i.e. the score of the bioresponse competition.
#    """
#    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
#    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    # X, y, X_submission = load_data.load()
    
    #load train and test 
    train_s  = pd.read_csv('../train.csv', index_col=0)
    test_s  = pd.read_csv('../test.csv', index_col=0)
    test_ind = test_s.index


    y = train_s.Hazard
    y = np.array(y)
    print np.any(np.isnan(y))
    train_s.drop('Hazard', axis=1, inplace=True)

    train_s.drop('T2_V10', axis=1, inplace=True)
    train_s.drop('T2_V7', axis=1, inplace=True)
    train_s.drop('T1_V13', axis=1, inplace=True)
    train_s.drop('T1_V10', axis=1, inplace=True)

    test_s.drop('T2_V10', axis=1, inplace=True)
    test_s.drop('T2_V7', axis=1, inplace=True)
    test_s.drop('T1_V13', axis=1, inplace=True)
    test_s.drop('T1_V10', axis=1, inplace=True)

    #columns = train.columns
    #test_ind = test.index


    train_s = np.array(train_s)
    test_s = np.array(test_s)
    
    # label encode the categorical variables
    for i in range(train_s.shape[1]):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])

    X = train_s
    X_submission = test_s

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(KFold(y.size, n_folds))

    clfs = [RandomForestRegressor(n_estimators=100, n_jobs=1),
            #RandomForestRegressor(n_estimators=100, n_jobs=1, criterion='entropy'),
            GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test) 
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission) 
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = linear_model.LinearRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test) 

    #print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print y_submission
    print "Saving Results."
    #np.savetxt(fname='testblah.csv', X=y_submission, fmt='%0.9f')
    
    #generate solution
    preds = pd.DataFrame({"Id": test_ind, "Hazard": y_submission})
    preds = preds.set_index('Id')
    preds.to_csv('blendtest.csv')
