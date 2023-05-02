import numpy as np
import pandas as pd
import scipy
import wittgenstein3 as lw3
import wittgenstein as lw
from sklearn.model_selection import cross_val_score

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

def acc_rate(df, class_feat, pos_class, cv=10, n_rep=10, W=0.5):
    
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
    
    X = df.loc[:,X_train.columns != class_feat]
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = X_train[class_feat]
    y = y.map(lambda x: 1 if x==pos_class else 0)
    
    # Define how to select the best element in the interval's boundary
    
    if param == 'W':
        interval = [0.1, 0.9]
        
        def return_best_a(interval):
            a = interval[0]
            b = interval[1]

            # Compute the elements' score and compare them
            ripper_clf = lw3.RIPPER(k=2, W = a)
            score_a = cross_val_score(ripper_clf, X, y, cv = cv) 
            
            ripper_clf = lw.RIPPER(k=2, W = b)
            score_b = cross_val_score(ripper_clf, X, y, cv = cv) 
            
            return interval[np.argmax([score_a, score_b])]
        
        
        # Now run iteratively the function above
        i = 1
        c = 1/2
        
        while i <= budget:
            
            i += 1
            best_point = return_best_a(interval)
            interval = [c, best_point].sort()
            c = (c + best_point)/2
            
        return best_point


        
        
        
        
        
        
        
        
    
    