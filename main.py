import os
import warnings

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor

from evaluate import evaluate, cv_coefs
from figures import plot_box

# Define random seed
SEED = 42

# Define the base folder
# PATH = '/home/erf6575/Documents/regression_wine/'
PATH = os.getcwd() + "/"

# Put extracted UCI data into baseFolder/data
white = pd.read_csv(PATH+'data/winequality-white.csv', sep=';')
red = pd.read_csv(PATH+'data/winequality-red.csv', sep=';')

# Define the feature and response columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
response = ['quality']


############ WHITE WINE ############

# Splitting and scaling data
x_train_white, x_test_white, y_train_white, y_test_white = train_test_split(white[features], white[response], test_size=0.30, random_state=SEED)

scaler = StandardScaler()
scaler.fit(x_train_white)
x_train_white = scaler.transform(x_train_white)
x_test_white = scaler.transform(x_test_white)
x_white =  scaler.transform(white[features])


lr_white = LinearRegression()
lr_white.fit(x_train_white, y_train_white)
mse_lr_white, r2_lr_white = evaluate(lr_white, x_test_white, y_test_white)
print(f"LinearRegression on White Wine: MSE ({mse_lr_white}), R^2({r2_lr_white})")

br_white = BayesianRidge()
br_white.fit(x_train_white, y_train_white)
mse_br_white, r2_br_white = evaluate(br_white, x_test_white, y_test_white)
print(f"BayesianRidge on White Wine: MSE ({mse_br_white}), R^2({r2_br_white})")

sgd_white = SGDRegressor()
sgd_white.fit(x_train_white, y_train_white)
mse_sgd_white, r2_sgd_white = evaluate(sgd_white, x_test_white, y_test_white)
print(f"SGDRegressor on White Wine: MSE ({mse_sgd_white}), R^2({r2_sgd_white})")


############ RED WINE ############

# Splitting and scaling data
x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(red[features], red[response], test_size=0.30, random_state=SEED)

scaler = StandardScaler()
scaler.fit(x_train_red)
x_train_red = scaler.transform(x_train_red)
x_test_red = scaler.transform(x_test_red)
x_red =  scaler.transform(red[features])


lr_red = LinearRegression()
lr_red.fit(x_train_red, y_train_red)
mse_lr_red, r2_lr_red = evaluate(lr_red, x_test_red, y_test_red)
print(f"LinearRegression on Red Wine: MSE ({mse_lr_red}), R^2({r2_lr_red})")

br_red = BayesianRidge()
br_red.fit(x_train_red, y_train_red)
mse_br_red, r2_br_red = evaluate(br_red, x_test_red, y_test_red)
print(f"BayesianRidge on Red Wine: MSE ({mse_br_red}), R^2({r2_br_red})")

sgd_red = SGDRegressor()
sgd_red.fit(x_train_red, y_train_red)
mse_sgd_red, r2_sgd_red = evaluate(sgd_red, x_test_red, y_test_red)
print(f"SGDRegressor on Red Wine: MSE ({mse_sgd_red}), R^2({r2_sgd_red})")


############ Comparing Coefficients by Wine Type ############

# Ingore the warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    np.seterr(all='ignore')
    
    cv_lr_white, coefs_lr_white = cv_coefs(lr_white, x_white, white[response], features, SEED)
    cv_br_white, coefs_br_white = cv_coefs(br_white, x_white, white[response], features, SEED)
    cv_sgd_white, coefs_sgd_white = cv_coefs(sgd_white, x_white, white[response], features, SEED)
    
    v_lr_red, coefs_lr_red = cv_coefs(lr_red, x_red, red[response], features, SEED)
    cv_br_red, coefs_br_red = cv_coefs(br_red, x_red, red[response], features, SEED)
    cv_sgd_red, coefs_sgd_red = cv_coefs(sgd_red, x_red, red[response], features, SEED)

# Check coefficients for white and red features 
# LinearRegression
plot_box(coefs_lr_white, coefs_lr_red, 
        title="LinearRegression Coefficients by Wine Type", 
        save_loc=PATH+"figures/coefs_lr")
# BayesianRidge
plot_box(coefs_br_white, coefs_br_red, 
        title="BayesianRidge Coefficients by Wine Type", 
        save_loc=PATH+"figures/coefs_br")
# SGDRegressor
plot_box(coefs_sgd_white, coefs_sgd_red, 
        title="SGDRegressor Coefficients by Wine Type", 
        save_loc=PATH+"figures/coefs_sgd")



