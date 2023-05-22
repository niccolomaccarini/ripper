import numpy as np
import pandas as pd
import scipy
import wittgenstein3 as lw4
import wittgenstein as lw
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, cross_validate, train_test_split
from scipy.stats import entropy
import math

def accuracy(df, class_feat, pos_class, cv=5, n_rep=10, W=0.5, metrics = ['accuracy']):
    
    """Function to rapidly compute the accuracy of RIPPERk using k-fold cross-validation,
    the test will be repeated n_rep times ans then the average accuracy will be returned
    
    metrics : list of metrics used to evaluate the accuracy, if it has only one element the function
    will return a float estimating the metric. If the list is longer than 1 the function will return a dictionary
    with all desired measures in the order given in the input. The options are:
    'accuracy' , 'f1' , 'roc_auc' respectively for standard, F1 and area under the curve scores.
    """
    
    # First dummify your categorical features and booleanize your class values to make sklearn happy

    X = df.loc[:,df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
                
    acc = np.zeros((n_rep, len(metrics)))
    
    for i in range(n_rep):
        ripper_clf = lw4.RIPPER(k=2, W=W)
        scores = cross_validate(ripper_clf, X, y, scoring = metrics, cv = cv)
        acc[i,:] = [np.mean(scores['test_' + metric]) for metric in metrics]
        
    acc = np.mean(acc, axis = 0)
    output = list(acc)
    
    if len(metrics) > 1:
        return dict(zip(metrics, output))
    
    return output[0]
                

def acc_rate(df, class_feat, pos_class, cv=5, n_rep=10, W=0.5, metrics = ['accuracy']):
    
    """Function to compare the accuracies of standard RIPPERk and enhanced RIPPERk on a given dataset df: first 
    the accuracies of both models acc_standard and acc_enhanced are computed, then the fraction 
                                     acc_enhanced / acc_standard
    is returned. """
    
    # First dummify your categorical features and booleanize your class values to make sklearn happy

    X = df.loc[:,df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
                
    acc_standard = np.zeros((n_rep, len(metrics)))
    
    for i in range(n_rep):
        ripper_clf = lw.RIPPER(k=2)
        scores = cross_validate(ripper_clf, X, y, scoring = metrics, cv = cv)
        acc_standard[i,:] = [np.mean(scores['test_' + metric]) for metric in metrics]
        
    acc_standard = np.mean(acc_standard, axis = 0)    
    acc_enhanced = accuracy(df, class_feat, pos_class, cv=cv, n_rep=n_rep, W=W)
    
    if len(metrics) > 1:
        return dict(zip(metrics, list(acc_enhanced/acc_standard)))
    
    return (acc_enhanced / acc_standard)[0]

    
    
def param_selection(df,
                    class_feat, 
                    pos_class = 1, 
                    cv = 5, 
                    param = 'W',
                    W = None,
                    budget = 15,
                    grid_search = True,
                    verbosity = False,
                    return_model = False):
    
    """Function to select the best hyperparameter using k-fold cross validation, as input it needs
    a training set with a given class_feat to classify. It can be used to select both W and the number of discretization bins.
    """
    
    # Preprocess the data as usual
    
    X = df.loc[:, df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
    
    # Define how to select the best element in the interval's 1/4th and 3/4th
    
    if param == 'W':
        interval = [0,1]
        
        if grid_search == False:
        
            def return_best_a(interval):
                low = interval[0]
                high = interval[1]

                a = 3*low/4 + high/4
                b = low/4 + 3*high/4

                candidates = [a, b]

                # Compute the elements' score and compare them
                ripper_clf = lw4.RIPPER(k=2, W = a)
                scores_a = cross_val_score(ripper_clf, X, y, cv = cv) 

                ripper_clf = lw4.RIPPER(k=2, W = b)
                scores_b = cross_val_score(ripper_clf, X, y, cv = cv) 

                return candidates[np.argmax([np.mean(scores_a), np.mean(scores_b)])]


            # Now run iteratively the function above
            i = 1
            p = 2

            while i <= budget:

                i += 1
                best_point = return_best_a(interval)
                interval = [best_point - (1/2)**p, best_point + (1/2)**p]
            p += 1
            
        else:    
        # Perform grid search to find the best value for W
        
            ws = np.arange(0.1, 1, 0.1)
            scores = np.zeros((10, len(ws)))
            models = np.zeros((10, len(ws)), dtype = np.object_)
            
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                for j in range(len(ws)):
                    # numpy.arange might make some floating point errors so we need to round the value of W after the first two digits
                    w = ws[j]
                    w = round(w, 2)

                    ripper_clf = lw4.RIPPER(k = 2, W = w)
                    ripper_clf.fit(pd.concat([X_train,y_train], axis=1), class_feat = class_feat)
                    scores[i, j] = ripper_clf.score(X_test,y_test)
                    models[i, j] = ripper_clf
 
            scores_avg = np.mean(scores, axis = 0)
            best_score_index = np.argmax(scores_avg)
            best_point = round(ws[best_score_index], 2)
            best_model = models[np.argmax(scores[:, best_score_index]), best_score_index]
                    
        if verbosity:
            print('Best W found: ' + str(best_point) + ' , best score: ' + str(best_score))
            
        if return_model:
            return best_point, best_model
            
        return best_point


        
def acc_rate_with_param_selection(df,
                                  class_feat, 
                                  pos_class, 
                                  cv_outer = 10,
                                  cv_inner = 5,
                                  param = 'W',
                                  W = None,
                                  budget = 15,
                                  metrics = [accuracy_score],
                                  n_rep = 10):
    
    """Function to compute the accuracy rate of improved RIPPERk using optimized W and standard RIPPERk. The idea is to 
    select via nested cross validation on X the best W, in the inner loop. Then compare the two
    algorithms on the test set and return the scorings rate. In this function the metrics' names don't have to be
    passed as strings."""
    
    # First dummify your categorical features and booleanize your class values to make sklearn happy
    X = df.loc[:,df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
                
    # Set the parameter grid for W and define an array to contain all scores
    acc_standard = np.zeros((n_rep, len(metrics)))
    acc_improved = np.zeros((n_rep, len(metrics)))
    p_grid = np.arange(0.1, 1, 0.1)

    for i in range(n_rep):
        cv_out = StratifiedKFold(cv_outer)
        cv_in = StratifiedKFold(cv_inner)
        scores_run_stand = np.zeros((len(metrics), cv_out.n_splits))
        scores_run = np.zeros((len(metrics), cv_out.n_splits))
        k = 0
                              
        for train_ix, test_ix in cv_out.split(X, y):
            # Split data
            X_train, X_test = X.loc[train_ix, :], X.loc[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # Define the models, train standard RIPPERk on the training set
            ripper_standard = lw.RIPPER(k=2)
            ripper_standard.fit(pd.concat([X_train,y_train], axis=1), class_feat = class_feat)

            # Execute search for best W and return the best model too
            best_W, best_model = param_selection(df = pd.concat([X_train,y_train], axis=1), 
                                                    class_feat = class_feat, 
                                                    return_model = True,
                                                    cv = cv_in)

            # Predict the values of the test set and compute scores
            y_hat_stand = ripper_standard.predict(X_test)
            y_hat = best_model.predict(X_test)  

            for j in range(len(metrics)):
                try:
                    metric = metrics[j]
                    scores_run_stand[j,k] = metric(y_hat_stand, y_test)
                    scores_run[j,k] = metric(y_hat, y_test)
                except:
                    print(str(metrics[j]) + ' not defined in this case.')

            k += 1
                              
                              
        # store the result
        acc_standard[i,] = np.mean(scores_run_stand, axis = 1)
        acc_improved[i,] = np.mean(scores_run, axis = 1)
                              
                              
    acc_rate = acc_improved / acc_standard
    output = np.mean(acc_rate, axis = 0)
    labels = [str(metric)[9:] for metric in metrics]
    
    if len(metrics) == 1:
        return output
    
    return dict(zip(labels, output))

def normalized_entropy(df = None, y = None, class_feat = None, pos_class = None):
    
    '''Function to compute the Shannon entropy of a given vector of probabilities. It is built on scipy.stats.entropy
    but adding the faculty of working with string vectors or datasets with a given class feature. Results are normalized 
    by dividing for the logarithm of the length of y.'''
    
    if y is not None:
        # If y is not numeric, then one needs to pass a positive class as input
        if pos_class:
            y = y.map(lambda x: 1 if x==pos_class else 0)
            
        return entropy(y) / math.log2(len(y))
    
    else:
        # When y is not given as input, then df, class_feat and pos_class must be
        y = df[class_feat]
        y = y.map(lambda x: 1 if x==pos_class else 0)
        
        return entropy(y) / math.log2(len(y))
    
def max_n_categories(df):
    '''Function to compute the number of possible categories for every variable of the dataset. Given a pandas
    dataframe it will return the maximum number of instances contained in every column.'''
        
    def count_elements(vector):
        return len(vector.unique())
    counts = df.apply(count_elements, axis = 0)
    
    return np.max(counts)                 