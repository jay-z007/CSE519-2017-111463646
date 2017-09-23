import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from math import log





def main():
    merged = pd.read_csv('./data/merge_copy.csv',index_col =0)


    logerror = merged['logerror']
    X_train, X_test, y_train, y_test = train_test_split(merged.drop(['logerror','parcelid'],axis=1), logerror, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(-1,1)) 
    s_train = scaler.fit_transform(X_train)
    s_test = MinMaxScaler(feature_range=(-1,1)).fit_transform(X_test)


    del merged
    del logerror
    del scaler

    knr = KNR(n_jobs=-1)
    knr_param_grid = {
        'n_neighbors' : [1500,1800,2500],
        'leaf_size': [100,200,300],
        'algorithm': ('ball_tree', 'kd_tree', 'brute'),
        
        
    }
    knr_gcv = GridSearchCV(knr,knr_param_grid,scoring = make_scorer(mean_absolute_error,greater_is_better=False))


    knr_gcv.fit(X_train,y_train)

    knr_best = knr_gcv.best_estimator_

    print "For unscaled features, mean_absolute_error is ", mean_absolute_error(y_test,knr_best.predict(X_test))
    print knr_best

    del knr_gcv
    del knr_best

    knr_gcv = GridSearchCV(knr,knr_param_grid,scoring = make_scorer(mean_absolute_error,greater_is_better=False))
    knr_gcv.fit(s_train,y_train)

    knr_best = knr_gcv.best_estimator_


    print "\n\n\nFor scaled features, mean_absolute_error is ", mean_absolute_error(y_test,knr_best.predict(s_test))
    print knr_best



    svr = svm.LinearSVR(tol=0.0000000001,max_iter = 2147483647)
    svr_param_grid = {
        'C' : [0.1,1,10,100,1000],
        'loss': ('epsilon_insensitive','squared_epsilon_insensitive')    
    }

    svr_gcv = GridSearchCV(svr,svr_param_grid,scoring = make_scorer(mean_absolute_error,greater_is_better=False),n_jobs=-1)


    ###SVR needs scaled data to perform well so we skip the unscaled part
    # svr_gcv.fit(X_train,y_train)


    # svr_best = svr_gcv.best_estimator_

    # print "\n\n\nFor scaled features, the mean_absolute_error is ", mean_absolute_error(y_test,svr_best.predict(X_test))
    # print svr_best



    svr_gcv.fit(s_train,y_train)


    svr_best = svr_gcv.best_estimator_

    print "\n\n\nFor scaled features, the mean_absolute_error is ", mean_absolute_error(y_test,svr_best.predict(s_test))
    print svr_best

if __name__ == '__main__':
    main()

