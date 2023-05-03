import numpy as np
import pandas as pd
import scipy
import wittgenstein3 as lw3
import wittgenstein as lw
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def accuracy(df, class_feat, pos_class, cv=10, n_rep=10, W=0.5):
    
    """Function to rapidly compute the accuracy of RIPPERk using k-fold cross-validation,
    the test will be repeated n_rep times ans then the average accuracy will be returned"""
    
    # First dummify your categorical features and booleanize your class values to make sklearn happy

    X = df.loc[:,df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
    
    
    acc = []

    for i in range(n_rep):
        ripper_clf = lw3.RIPPER(k=2, W=W)
        scores = cross_val_score(ripper_clf, X, y, cv = 10) 
        acc += [scores]

    return np.mean(acc)

def acc_rate(df_train, df_test, class_feat, pos_class, cv=10, n_rep=10, W=0.5):
    
    """Function to compare the accuracies of standard RIPPERk and enhanced RIPPERk on a given dataset df: first 
    the accuracies of both models acc_standard and acc_enhanced are computed, then the fraction 
                                     acc_enhanced / acc_standard
    is returned. """
    
    # Compute the accuracy of standard RIPPERk
    
    X = df.loc[:,df.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = df[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
    
    acc_standard = []

    for i in range(n_rep):
        ripper_clf = lw.RIPPER(k=2)
        scores = cross_val_score(ripper_clf, X, y, cv = 10) 
        acc_standard += [scores]
        
    acc_enhanced = accuracy(df, class_feat, pos_class, cv=cv, n_rep=n_rep, W=W)   
    
    return acc_enhanced / acc_standard

    
    
def param_selection(X_train, class_feat, pos_class, cv = 10, param = 'W', W = None, budget = 15):
    
    """Function to select the best hyperparameter using k-fold cross validation, as input it needs
    a training set with a given class_feat to classify. It can be used to select both W and the number of discretization bins"""
    
    # Preprocess the data as usual
    
    X = X_train.loc[:,X_train.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = X_train[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
    
    # Define how to select the best element in the interval's 1/4th and 3/4th
    
    if param == 'W':
        interval = [0,1]
        
        def return_best_a(interval):
            low = interval[0]
            high = interval[1]
            
            a = 3*low/4 + high/4
            b = low/4 + 3*high/4
            
            candidates = [a, b]

            # Compute the elements' score and compare them
            ripper_clf = lw3.RIPPER(k=2, W = a)
            scores_a = cross_val_score(ripper_clf, X, y, cv = cv) 
            
            ripper_clf = lw3.RIPPER(k=2, W = b)
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
            
        return best_point


        
def acc_rate_with_param_selection(df, class_feat, pos_class, cv = 10, param = 'W', W = None, budget = 15):
    
    """Function to compute the accuracy rate of improved RIPPERk using optimized W and standard RIPPERk. The idea is to
    separate the dataset in a training and test set, and select via cross validation on X_train the best W. Then compare the two
    algorithms on the test set and return the accuracy rate."""
    
    # Separate the data in training and test set
    
    X_train, X_test = train_test_split(df, test_size = 0.25)
    
    # Find the best value for W
    
    W = param_selection(X_train, class_feat, pos_class, cv = cv, param = 'W', W = None, budget = budget)
    
    # Train the algorithms
    
    ripper_standard = lw.RIPPER(k=2)
    ripper_improved = lw3.RIPPER(k=2, W=W)
    
    ripper_standard.fit(X_train, class_feat = class_feat, pos_class = pos_class)
    y_test = X_test[class_feat]
    acc_standard = ripper_standard.score(X_test, y_test)
    
    ripper_improved.fit(X_train, class_feat = class_feat, pos_class = pos_class)
    y_test = X_test[class_feat]
    acc_improved = ripper_improved.score(X_test, y_test)
        
    return {'acc_rate':acc_improved / acc_standard, 'acc_standard': acc_standard, 'acc_improved': acc_improved}
        
        
        
        
        
        
    
    