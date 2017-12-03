import numpy as np
import pandas as pd
import seaborn as sbn
import pickle as pkl
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,make_scorer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics
from math import log


import pdb; pdb.set_trace()
def eval_probs(y_prob,y_test, labels=[-1,0,1],show_plot=False):
    """
    y_prob : predict_proba output of a model
    labels : all unique labels in the classification
    """
    vals = []
    titles = {}
    titles[-1] = "Away Win"
    titles[0] = "Draw"
    titles[1] = "Home Win"

    y_prob = pd.DataFrame(y_prob, columns=[-1, 0, 1])
    # y_prob = y_prob.round(1)
    y_prob['label'] = y_test.reset_index(drop=True)
    slopes= []
    
    for ind, label in enumerate(labels):
        vals = sorted(y_prob[label].unique())
        
        y_col = []
        
        
        for i in vals:
            tot_len = y_prob[(y_prob[label] == i)].shape[0]
            matching_len = y_prob[(y_prob[label] == i) & (y_prob['label'] == label)].shape[0]
            y_col.append(matching_len*1.0/tot_len)
        
        slopes.append(np.polyfit(vals,y_col,deg=1)[0])    

        if show_plot:
            plt.figure(figsize=(7,21))
            plt.subplot(311+ind)
            plt.bar(vals, y_col, width=0.05)
            plt.title(titles[label], fontsize=16)
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=15)
            plt.yticks(np.arange(0.0, 1.1, 0.1),fontsize=15)
            plt.show()
    return mean_squared_error([1,1,1],slopes)**0.5
	
def save_model(model, model_name):
	saved_models_root = "./saved_models/" 
	filename = saved_models_root + model_name + datetime.now().strftime('%m%d_%H%M%S') + ".pkl"
												 
	with open(filename, 'wb') as outfile:
		pkl.dump(model, outfile)


def main():
	# with open("./data/train/train_data.pkl") as infile:
	# 	train_data = pkl.load(infile)

	# columns = ['date', 'B365H', 'B365D', 'B365A', 'match_id', 'home_team', 'away_team', 
	# 		'winner', 'minute', 'H_Goal', 'A_Goal']
	# train_data = train_data[columns]

	# train_data.drop_duplicates(inplace=True, keep='first')

	# X = train_data.drop(['winner', 'date', 'match_id', 'home_team', 'away_team'], axis=1)
	# Y = train_data['winner']

	# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


	# lr = LogisticRegression(n_jobs=-1,multi_class='multinomial',solver='lbfgs')
	# lr.fit(X_train,y_train)

	# y_pred = lr.predict(X_test)

	# print metrics.accuracy_score(y_test, y_pred)

	# y_prob = lr.predict_proba(X_test)

	with open('y_prob.pkl') as f:
		y_prob = pkl.load(f)

	with open('y_test.pkl') as f:
		y_test = pkl.load(f)

	

	print eval_probs(y_prob, y_test, [-1,0,1],show_plot=True)

if __name__ == '__main__':
	main()


