import numpy as np
import pandas as pd
from sklearn import linear_model, cross_validation
import math
import csv
from sklearn import svm
import matplotlib.pyplot as plt

xticks = ['', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
          'chlorides', 'free_slfr_diox', 'total_slfr_diox', 'density', 'pH', 'sulphates', 'alcohol', '']

def create_alpha_values():
    """
    Creates the list of alpha vlaues to use from 1 to 1000 by 0.1 
    """
    alpha_np = np.arange(1, 1000, 0.1)
    alpha_lst = alpha_np.tolist()
    for i in range(len(alpha_lst)):
        alpha_lst[i] = round(alpha_lst[i], 1)
    return alpha_lst 

def pickPoints(data):
    chosen = []
    size = len(data)
    k = max(1, size - (size//10))
    while len(chosen) < k:
        num = int(np.random.uniform()*size)
        if num not in chosen and num != 0 and num != 1599:
            chosen.append(num)
    return chosen

def create_training_matrix(chosen):
    training_matrix_list = []
    training_y_values = []
    for val in chosen:
        current_row = []
        for j in range(len(column_names) - 1):
            current_row.append(data_matrix[column_names[j]][val])
        training_matrix_list.append(current_row)
        training_y_values.append(data_matrix['quality'][val])
    return training_matrix_list, training_y_values 

def create_test_matrix(chosen):
    test_matrix_list = [] 
    test_y_values = [] 
    for i in range(len(data_matrix['quality'])):
        current_row = []
        if i not in chosen:
            for j in range(len(column_names) - 1):
                current_row.append(data_matrix[column_names[j]][i])
            test_matrix_list.append(current_row)
            test_y_values.append(data_matrix['quality'][i])
    return test_matrix_list, test_y_values

def manually_calculate_errors(test_matrix_list, test_y_values, training_matrix_list, training_y_values, alphas):
    error = []
    for alpha in alphas:
        clf = linear_model.ElasticNet(fit_intercept = True, alpha = alpha)
        clf.fit(training_matrix_list, training_y_values)
        coefficients, intercept = clf.coef_.tolist(), clf.intercept_
        local_error = 0 
        for i in range(len(test_matrix_list)):
            curr_row = test_matrix_list[i]
            curr_y = test_y_values[i]
            row_val = 0 
            for j in range(len(curr_row)):
                row_val += curr_row[j] * coefficients[j]
            row_val += intercept 
            local_error += abs(curr_y - row_val) ** 2
        error.append(float(local_error) / len(test_matrix_list))
    return min(error), alphas[error.index(min(error))], error 

### MANUAL DATA SCRUBBING END ### 

def clean_column_names(columns):
    """
    Cleans the column names to remove quotes and spaces
    """
    new_col_names = []
    for name in columns:
        new_col_names.append(name.replace('"', '').replace('\n', ''))
    return new_col_names

def setup_original_matrixes(columns):
    """
    Used to populate the column names of the matrix to record all data and assign a number 
    to each column name for easier bookkeeping later. 
    """
    data_matrix  = {}
    column_names = {}
    for i in range(len(columns)):
        column_names[i] = columns[i]
        data_matrix[columns[i]] = [] 
    return data_matrix, column_names

def create_OLS_model(features, values, xticks):
    """
    Creates the OLS model given the features and y-values
    """
    clf = linear_model.LinearRegression(fit_intercept = True)
    clf.fit(features, values)
    plot_coefficients_against_features(xticks, clf.coef_.tolist())

def create_ridge_model(features, values, alpha):
    """
    Creates the Ridge model given the features, y-values, and alpha
    """
    clf = linear_model.Ridge(fit_intercept = True, alpha = alpha)
    clf.fit(features, values)
    plot_coefficients_against_features(xticks, clf.coef_.tolist())

def create_lasso_model(features, values, alpha):
    """
    Creates the Lasso model given the features, y-values, and alpha
    """
    clf = linear_model.Lasso(fit_intercept = True, alpha = alpha)
    clf.fit(features, values)
    plot_coefficients_against_features(xticks, clf.coef_.tolist())

def create_elastic_net_model(features, values, alpha):
    """
    Creates the Elastic Net model give the features, y-values, and alpha
    """
    clf = linear_model.ElasticNet(fit_intercept = True, alpha = alpha)
    clf.fit(features, values)
    plot_coefficients_against_features(xticks, clf.coef_.tolist())

def plot_coefficients_against_features(xticks, coefficients):
    """
    PART 2 
    Given coefficients, plots their values against the set of features 
    """
    x_vals = [i * 3 for i in range(13)]
    coefficients.insert(0, 0)
    coefficients.append(0)
    for i in range(len(coefficients)):
        coefficients[i] = round(coefficients[i], 3)
    plt.xticks(x_vals, xticks)
    plt.plot(x_vals, coefficients, 'ro')
    for i in range(len(x_vals)):
        if i != 0 and i != len(x_vals) - 1:
            plt.text(x_vals[i] + 0.2, coefficients[i] - 0.2, coefficients[i])
    plt.show()

def plot_tuning_parameter_errors(tuning_parameters, errors):
    plt.xlabel("tuning parameter value")
    plt.ylabel('error')
    plt.plot(tuning_parameters, errors)
    plt.axis([0,10.1,0,1])
    plt.show()

def calculate_tuning_parameter_errors(fit_type, features, values, alpha_vals):
    """
    Enumerates through 10,000 alpha values to find the elbow for the tuning parameters
    """
    calculated_errors = []
    for alpha_val in alpha_vals:
        if fit_type == 'ridge':
            fit = linear_model.Ridge(fit_intercept = True, alpha = alpha_val)
        elif fit_type == 'lasso':
            fit = linear_model.Lasso(fit_intercept = True, alpha = alpha_val)
        else:
            fit = linear_model.ElasticNet(fit_intercept = True, alpha = alpha_val)
        k_fold = cross_validation.KFold(1400, n_folds=10)
        score = cross_validation.cross_val_score(fit, features, values, cv=k_fold, scoring='mean_squared_error')
        total = 0 
        for entry in score:
            if entry < 0:
                total += abs(entry)
            else:
                total += entry
        calculated_errors.append(total)
        print alpha_val, sum(score) * -1
    return calculated_errors

with open("winequality-red.csv") as f:
    wine_quality_data = f.readlines()

columns = clean_column_names(wine_quality_data[0].split(";"))
data_matrix, column_names = setup_original_matrixes(columns)
alpha_values, y = create_alpha_values(), data_matrix['quality']

for data in wine_quality_data[1:]:
    values = data.split(";")
    for i in range(len(values)):
        data_matrix[column_names[i]].append(float(values[i]))
matrix_list = []

for i in range(len(data_matrix['quality'])):
    current_row = []
    for j in range(len(column_names) - 1):
        current_row.append(data_matrix[column_names[j]][i])
    matrix_list.append(current_row)

chosen = pickPoints(wine_quality_data)
training_matrix, training_y = create_training_matrix(chosen)
test_matrix, test_y = create_test_matrix(chosen)