'''
Final Pipeline
'''

import pandas as pd
import numpy as np
import csv
import sklearn.tree as tree
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
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split


#Functions used in go file:
def read_data(filename):
	'''
	Read csv file to a pandas dataframe
	Inputs: filename (str)
	Returns: pandas dataframe
	'''
	data = pd.read_csv(filename)
	return data


def temporal_validate(start_year, end_year, prediction_windows):
	'''
	Create temporal splits for data
	Inputs: start_year, end_year, prediction_windows
	Returns: splits - dates based on inputs above
	'''
	splits = []
	start_time_date = start_year
	end_time_date = end_year

	for prediction_window in prediction_windows:
		on_window = 1
		test_end_time = start_time_date
		while (end_time_date > test_end_time):
			train_start_time = start_time_date
			train_end_time = train_start_time + on_window * prediction_window - 1
			test_start_time = train_start_time + on_window * prediction_window
			test_end_time = test_start_time + prediction_window - 1
			trains = (train_start_time, train_end_time)
			tests = (test_start_time, test_end_time)
			splits.append([trains, tests])
			on_window += 1
	return splits


def temp_val_split(df, date_col, date_1, date_2):
	'''
	Create temp val train df
	Inputs: df, date_col, date_1, date_2
	Returns: temp val df
	'''
	return df[df[date_col].between(date_1, date_2, inclusive=True)]


def load_dfs(df, datecol, lst_train_test_dates):
	'''
	Create training and testing dataframes based on the temporal splits
	Inputs: dataframe, date column, list of dates
	Returns: list of dataframes
	'''
	final_lst = []
	for datepair in lst_train_test_dates:
		train_df = temp_val_split(df, datecol, datepair[0][0], datepair[0][1])
		test_df = temp_val_split(df, datecol, datepair[1][0], datepair[1][1])
		final_lst.append((train_df, test_df))
	return final_lst


def find_top(df, datecol, outcome_col, master_dic, thres):
	'''
	Find top % of a column in a df, create set, add results to dictionary
	Inputs: df, datecol, outcome_col, master_dic, thres
	Returns: None, updates master_dic with set results
	'''
	top = set()
	for year in df[datecol].unique():
		year_slice = df[df[datecol] == year]
		top.update(set(year_slice.sort_values(by=[outcome_col], ascending=False).head(int(len(year_slice)*thres))['GEOID']))
	for year in df[datecol].unique():
		master_dic[year] = top
	return None


def create_binary(row, master_dic):
	'''
	Create outcome variable
	Inputs: row, master dic
	Returns: result
	'''
	if row['GEOID'] in master_dic[row['Date']]:
		result = 1
	else:
		result = 0
	return result


def convert_to_categorical(df, cat_cols):
	'''
	Convert a row to categorical
	Inputs: df, cat_cols
	Returns: None, updates dataframe in place
	'''
	for col in cat_cols:
		df[col] = df[col].astype('category').cat.codes
	return


def fill_null_auto(df):
	'''
	Fill NAs in dataframe
	Input: df
	Returns: None, updates df in place
	'''
	cols = df.columns.tolist()
	for col in cols:
		if df[col].isnull().any():
			df[col].fillna(df[col].median(), inplace=True)
	return


def convert_tf_to_binary(df):
	'''
	Convert true/false to 1/0
	Inputs: df
	Returns: None, updates dataframe in place
	'''
	cols = df.columns.tolist()
	for col in cols:
		if is_string_dtype(df[col]):
			if (df[col] == 't').any():
				df[col] = df[col].map({'f': 0, 't': 1})
	return


def convert_to_disc(df, disc_cols, bins=4, labels=[0,1,2,3]):
	'''
	Discretize continuous variables into discrete variables
	Inputs: df, cols, bins, labels
	Returns: None, updates the columns in df
	'''
	for col in disc_cols:
		df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
	return


def clean_data(df_list, cat_cols=None, disc_cols=None):
	'''
	Clean and process data after train/test split
	Cleaning includes filling nulls, converting true/false to binaries, 
	converting columns to categorical, and continuous to discrete
	Inputs: 
	- df_list: list of test/control dataframes
	- cat_cols: list of columns to convert to categorical
	- disc_cols: list of columns to convert to discrete
	Returns: None, updates df in place for all cleaning
	'''
	for tup in df_list:
		for df in tup:
			df['median-gross-rent'] = df.groupby("GEOID")['median-gross-rent'].transform(lambda x: x.fillna(x.median()))
			df['median-household-income'] = df.groupby("GEOID")['median-household-income'].transform(lambda x: x.fillna(x.median()))
			df['median-property-value'] = df.groupby("GEOID")['median-property-value'].transform(lambda x: x.fillna(x.median()))
			df['rent-burden'] = df.groupby("GEOID")['rent-burden'].transform(lambda x: x.fillna(x.median()))
			df['avg_household_size_owner'] = df.groupby("GEOID")['avg_household_size_owner'].transform(lambda x: x.fillna(x.median()))
			df['avg_household_size_renter'] = df.groupby("GEOID")['avg_household_size_renter'].transform(lambda x: x.fillna(x.median()))
			convert_to_categorical(df, cat_cols)
			fill_null_auto(df)
			convert_tf_to_binary(df)
			convert_to_disc(df, disc_cols)
	return


#Additional pipeline functions:
#statistics
def describe_data(data, colname):
	'''
	Find the distribution of a variable in the dataset
	Inputs: data(dataframe), colname(str)
	Returns: data description
	'''
	return data[colname].describe()


def boxplot(data, colname):
	'''
	Make the boxplot of a variable in the dataset 
	Inputs: data(dataframe), colname(str)
	Returns: boxplot
	'''
	return data[colname].plot.box()


def density_plot(data, colname):
	'''
	Make the density plot of a variable in the dataset
	Inputs: data(dataframe), colname(str)
	Returns: density plot 
	'''
	return data[colname].plot.density()


def find_summaries(data, colnames):
	'''
	Find summaries of all variables that we are interested in
	Inputs: data(dataframe), colnames (list)
	Returns: summaries dataframe
	'''
	return data[colnames].describe()


def find_corr(data, col1, col2):
	'''
	Find correlations between variables
	Inputs: data(dataframe). col1(str), col2(str)
	Returns: correlation of two columns
	'''
	return data[col1].corr(data[col2])


#data cleaning and formatting
def convert_column_type(df, col, new_type):
	'''
	Convert a column to a new type 
	Inputs: df, column, new_type
	Returns: None, updates the column in df
	'''
	df[col] = df[col].astype(new_type)
	return


def convert_to_datetime(df, cols):
	'''
	'''
	for col in cols:
		df[col] =  pd.to_datetime(df[col])


def convert_type(data, colname, target_type):
	'''
	Convert datatype of a column
	Input: data (dataframe), colname (str), type (str)
	'''
	data[colname] = data[colname].astype(target_type)


def if_null(data, colname):
	'''
	Check if the column contains NULLs
	Inputs: data(dataframe)，colname(str)
	Returns: 
	'''
	return data[colname].isnull().values.any()


def discretize_col(data, columns, bins, labels):
	'''
	Discretize a set of columns in a dataset
	To discretize the continuous variable into three discrete variables: 0, 1, and 2;
	the boundaries are the minimum value, the 25% quantile, the 75% quantile, and the maximum value.
	Inputs: data, pandas dataframe
			columns, list
			bins, list
			labels, list
	Returns: None
	'''
	for column in columns:
		data[column] = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True)
	return


def fill_na(data, columns):
	'''
	Fill in NA values with mean
	Inputs: pandas dataframe, columns, list
	Returns: None, updates dataframe in place
	'''
	for column in columns:
		if data[column].isnull().any():
			data[column] = data[column].fillna(data[column].median())
	return


def label_to_dummy(item, bar):
	'''
	Convert the label to dummy variables
	Inputs: item, bar
	Returns: dummy result
	'''
	if item >= bar:
		result = 1
	else:
		result = 0
	return result


def to_dummy(data, column):
	'''
	Convert columns in categorical variables to dummy variables
	Inputs: data, pandas dataframe, column, list
	Returns: updated dataframe with dummy vars
	'''
	data = pd.get_dummies(data, columns=column)
	return data


def slice_time_data(data, column, start_time, end_time):
	'''
	Slice time series data by time
	Inputs: data, column, start_time, end_time
	Returns: sliced dataframe
	'''
	return data[(data[column] >= start_time) & (data[column] <= end_time)]


def extract_train_test(df_train, df_test, dep_var, pred_vars):
	'''
	Create train test splits
	Inputs: df_train, df_test, dep_var, pred_vars
	Returns: train test splits
	'''
	X_train = df_train[pred_vars]
	X_test = df_test[pred_vars]
	y_train = df_train[dep_var]
	y_test = df_test[dep_var]
	return X_train, X_test, y_train, y_test


def label_to_dummy(item, bar):
	'''
	item: int
	bar: int
	'''
	if item >= bar:
		result = 1
	else:
		result = 0
	return result


#The following code referenced from the following website: https://github.com/rayidghani/magicloops/blob/master/simpleloop.py, credit to Rayid Ghani
def define_clfs_params():
	clfs = {
		'BG': BaggingClassifier(n_estimators=10),
		'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
		'LR': LogisticRegression(penalty='l1', C=1e5),
		'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
		'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
		'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
		'DT': DecisionTreeClassifier(),
		'KNN': KNeighborsClassifier(n_neighbors=3),
		'NB': GaussianNB()}
	grid = {
	'BG': {'n_estimators': [10,50]},  
	'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
	'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
	'SVM' :{'C' :[0.01,0.1,1],'kernel':['linear']},
	'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.01,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,5,20]},
	'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
	'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
	'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
	'NB' : {}}
	return clfs, grid


def models_to_run():
	models_to_run = ['RF', 'DT', 'LR', 'NB', 'KNN', 'BG', 'AB', 'GB', 'SVM']
	return models_to_run
	

def joint_sort_descending(l1, l2):
	idx = np.argsort(l1)[::-1]
	return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
	cutoff_index = int(len(y_scores) * (k / 100.0))
	test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
	return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
	y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores, k)
	precision = precision_score(y_true, preds_at_k)
	return precision


def recall_at_k(y_true, y_scores, k):
	y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores, k)
	recall = recall_score(y_true, preds_at_k)
	return recall


def f1_at_k(y_true, y_scores, k):
	precision = precision_at_k(y_true, y_scores, k)
	recall = recall_at_k(y_true, y_scores, k)
	return 2 * (precision * recall)/(precision + recall)


def plot_precision_recall_n(y_true, y_prob, model_name):
	from sklearn.metrics import precision_recall_curve
	y_score = y_prob
	precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
	precision_curve = precision_curve[:-1]
	recall_curve = recall_curve[:-1]
	pct_above_per_thresh = []
	number_scored = len(y_score)
	for value in pr_thresholds:
		num_above_thresh = len(y_score[y_score>=value])
		pct_above_thresh = num_above_thresh / float(number_scored)
		pct_above_per_thresh.append(pct_above_thresh)
	pct_above_per_thresh = np.array(pct_above_per_thresh)
	plt.clf()
	fig, ax1 = plt.subplots()
	ax1.plot(pct_above_per_thresh, precision_curve, 'b')
	ax1.set_xlabel('percent of population')
	ax1.set_ylabel('precision', color='b')
	ax2 = ax1.twinx()
	ax2.plot(pct_above_per_thresh, recall_curve, 'r')
	ax2.set_ylabel('recall', color='r')
	ax1.set_ylim([0,1])
	ax1.set_ylim([0,1])
	ax2.set_xlim([0,1])
	name = model_name
	plt.title(name)
	plt.show()
	

def clf_loop_all_data(models_to_run, clfs, grid, train_test_dfs, pred_vars, dep_var, thresholds, csv_path):
	"""Runs the loop using models_to_run, clfs, gridm and the data
	"""
	with open(csv_path, 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter = ',')
		writer.writerow(['time_period', 'model_type', 'clf', 'parameters', 'threshold', 'auc-roc',
											'precision', 'recall', 'f1_score'])
		for i, tup in enumerate(train_test_dfs):
			train_df, test_df = tup
			X_train = train_df[pred_vars]
			y_train = train_df[dep_var]
			X_test = test_df[pred_vars]
			y_test = test_df[dep_var]
			period = "train_test_data_period" + "_" + str(i)
			print(period)
			for index, clf in enumerate([clfs[x] for x in models_to_run]):
				print(models_to_run[index])
				parameter_values = grid[models_to_run[index]]
				for p in ParameterGrid(parameter_values):
					clf.set_params(**p)
					if 'SVM' in models_to_run[index]:
						y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
					else:
						y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
					y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

					for threshold in thresholds:
						precision = precision_at_k(y_test_sorted, y_pred_probs_sorted, threshold)
						recall = recall_at_k(y_test_sorted, y_pred_probs_sorted, threshold)
						f1_score = f1_at_k(y_test_sorted, y_pred_probs_sorted, threshold)
							
						writer.writerow([period, models_to_run[index], clf, p, threshold,
														 roc_auc_score(y_test, y_pred_probs), precision,
														 recall, f1_score])	
	csv_file.close()
	return


def get_feature_importance(X_train, model):
	importances = model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model.estimators_],
			 axis=0)
	indices = np.argsort(importances)[::-1]
	# Print the feature ranking
	print("Feature ranking:")
	for f in range(X_train.shape[1]):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X_train.shape[1]), importances[indices],
		color="r", yerr=std[indices], align="center")
	plt.xticks(range(X_train.shape[1]), indices)
	plt.xlim([-1, X_train.shape[1]])
	plt.show()
	return