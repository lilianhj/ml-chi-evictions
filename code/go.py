'''
Go file to run Pipeline
'''

import final_pipeline_for_vm as fp
import numpy as np
import pandas as pd
import numpy as np
import csv
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from sklearn.metrics import *
from sklearn.model_selection import ParameterGrid
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split


models_to_run = ['RF', 'DT', 'LR', 'NB', 'KNN', 'BG', 'AB', 'GB', 'SVM']


def run(models_to_run):
	'''
	Go function to run pipeline
	Inputs: list of models to run
	Returns: None, runs the pipeline and outputs results to a csv file 
	'''
	#read data
	og_df = fp.read_data('raw_data/full_data_chicago.csv')

	#temporal validation
	lst_train_test_dates = fp.temporal_validate(2005, 2016, [3])

	#clean and filter data
	data_select = og_df[og_df['Date'] >= 2005]
	data_select['crime_rate'] = data_select['Sum']/data_select['population']
	data_select['less_hs_rate'] = data_select['Total_Less_Than_HS']/data_select['population']
	data_select['hs_grad_rate'] = data_select['Total_HS_Grad']/data_select['population']
	data_select['some_college_rate'] = data_select['Total_Some_College_or_AAS']/data_select['population']
	data_select['bachelors_rate'] = data_select['Total_Bachelors']/data_select['population']
	to_drop = ['Unnamed: 0', 'name', 'parent-location', 'holc_grade', 'holc_id', 'Unnamed: 0.1',
			  'eviction-filings', 'evictions', 'eviction-filing-rate', 'tract_id','Sum']
	data_select = data_select.drop(columns=to_drop)
	final_list = fp.load_dfs(data_select, 'Date', lst_train_test_dates)
	(train_1, test_1), (train_2, test_2), (train_3, test_3) = final_list
	master_dic = {}
	fp.find_top(train_1, 'Date', 'eviction-rate', master_dic, 0.1)
	fp.find_top(test_1, 'Date', 'eviction-rate', master_dic, 0.1)
	fp.find_top(test_2, 'Date', 'eviction-rate', master_dic, 0.1)
	fp.find_top(test_3, 'Date', 'eviction-rate', master_dic, 0.1)
	for tup in final_list:
		for df in tup:
			df['if_top_10'] = df.apply(lambda x: fp.create_binary(x, master_dic), axis=1)
	to_discretize = ['median-gross-rent', 'median-household-income', 'median-property-value']
	remain_dummy = ['low-flag', 'imputed', 'subbed']
	to_category = ['Primary Type']
	label = ['if_top_10']
	second_drop = ['eviction-rate']
	for data in final_list:
		for df in data:
			df.drop(columns=second_drop, inplace=True)
	for tup in final_list:
		for data in tup:
			data.replace(np.inf, np.nan, inplace=True)
			data.replace(-np.inf, np.nan, inplace=True)
	fp.clean_data(final_list, cat_cols=to_category, disc_cols=to_discretize)

	#set variables
	pred_vars = ['population', 'poverty-rate',
	   'renter-occupied-households', 'pct-renter-occupied',
	   'median-gross-rent', 'median-household-income', 'median-property-value',
	   'rent-burden', 'pct-white', 'pct-af-am', 'pct-hispanic', 'pct-am-ind',
	   'pct-asian', 'pct-nh-pi', 'pct-multiple', 'pct-other', 'low-flag',
	   'avg_household_size_owner',
	   'avg_household_size_renter', 'less_hs_rate', 'hs_grad_rate',
	   'some_college_rate', 'bachelors_rate', 'Mean_hours_worked',
	   'Primary Type', 'redlined', 'crime_rate']
	dep_var = 'if_top_10'
	clfs = {'AB': AdaBoostClassifier(algorithm='SAMME',
						base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
						max_features=None, max_leaf_nodes=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=1, min_samples_split=2,
						min_weight_fraction_leaf=0.0, presort=False, random_state=None,
						splitter='best'),
						learning_rate=1.0, n_estimators=200, random_state=None),
			'BG': BaggingClassifier(base_estimator=None, bootstrap=True,
						bootstrap_features=False, max_features=1.0, max_samples=1.0,
						n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
						verbose=0, warm_start=False),
			'DT': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
						max_features=None, max_leaf_nodes=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=1, min_samples_split=2,
						min_weight_fraction_leaf=0.0, presort=False, random_state=None,
						splitter='best'),
			'GB': GradientBoostingClassifier(criterion='friedman_mse', init=None,
						learning_rate=0.05, loss='deviance', max_depth=6,
						max_features=None, max_leaf_nodes=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=1, min_samples_split=2,
						min_weight_fraction_leaf=0.0, n_estimators=10,
						presort='auto', random_state=None, subsample=0.5, verbose=0,
						warm_start=False),
			'KNN': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
						metric_params=None, n_jobs=1, n_neighbors=3, p=2,
						weights='uniform'),
			'LR': LogisticRegression(C=100000.0, class_weight=None, dual=False,
						fit_intercept=True, intercept_scaling=1, max_iter=100,
						multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
						solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
			'NB': GaussianNB(priors=None),
			'RF': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
						max_depth=None, max_features='auto', max_leaf_nodes=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=1, min_samples_split=2,
						min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
						oob_score=False, random_state=None, verbose=0,
						warm_start=False),
			'SVM': svm.LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
	 						intercept_scaling=1, loss='squared_hinge', max_iter=1000,
multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)}
	grid = {'BG': {'n_estimators': [10,50]},  
			'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
			'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
			'SVM' : {'C' :[0.01,0.1,1],'penalty':['l1','l2']},
			'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.01,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,5,20]},
			'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
			'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
			'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
			'NB' : {}}

	train_3.to_csv("output_files/train.csv", index=False)
	test_3.to_csv("output_files/test.csv", index=False)

	#run models, output results
	fp.clf_loop_all_data(models_to_run, clfs, grid, final_list, pred_vars, dep_var, [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0], "results.csv")
	return


if __name__ == "__main__":
	run(models_to_run)