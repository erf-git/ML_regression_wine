import warnings

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

# Define random seed
SEED = 42

# Define the base folder
PATH = '/home/erf6575/Documents/regression_wine/'

# Put extracted UCI data into baseFolder/data
white = pd.read_csv(PATH+'data/winequality-white.csv', sep=';')
red = pd.read_csv(PATH+'data/winequality-red.csv', sep=';')

# Define the feature and response columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
response = ['quality']


############ FUNCTIONS ############

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    # Looking at mean squared errored and R^2
    mse = np.round(mean_squared_error(y_test, y_pred), decimals=5)
    r2 = np.round(r2_score(y_test, y_pred), decimals=5)
    
    return mse, r2

def cv_coefs(model, x, y):
    
    cv = cross_validate(
        model,
        x,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED),
        return_estimator=True,
        n_jobs=-1,
    )
    
    # print([model.coef_ for model in cv["estimator"]][0].shape)
    # print([model.coef_ for model in cv["estimator"]])
    
    # Ensure each array has shape (11,)
    arrays = [model.coef_ for model in cv["estimator"]]
    normalized = [a.ravel() for a in arrays]
    
    coefs = pd.DataFrame(
        np.vstack(normalized),
        columns=features,
    )
    
    return cv, coefs


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
    
    cv_lr_white, coefs_lr_white = cv_coefs(lr_white, x_white, white[response])
    cv_br_white, coefs_br_white = cv_coefs(br_white, x_white, white[response])
    cv_sgd_white, coefs_sgd_white = cv_coefs(sgd_white, x_white, white[response])
    
    v_lr_red, coefs_lr_red = cv_coefs(lr_red, x_red, red[response])
    cv_br_red, coefs_br_red = cv_coefs(br_red, x_red, red[response])
    cv_sgd_red, coefs_sgd_red = cv_coefs(sgd_red, x_red, red[response])


legend_elements = [
    Patch(facecolor="skyblue", label="White"),
    Patch(facecolor="tomato", label="Red")
]

plt.figure(figsize=(8, 6))
sns.boxplot(data=coefs_lr_white, orient="h", color="skyblue")
sns.boxplot(data=coefs_lr_red, orient="h", color="tomato")
plt.axvline(x=0, color="gray", ls="--")
plt.xlabel("Coefficients")
plt.title("LinearRegression Coefficients by Wine Type")
plt.legend(handles=legend_elements, title="Wine Type", loc="upper right")
plt.tight_layout()
plt.savefig(PATH+"figures/coefs_lr")
# plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=coefs_br_white, orient="h", color="skyblue")
sns.boxplot(data=coefs_br_red, orient="h", color="tomato")
plt.axvline(x=0, color="gray", ls="--")
plt.xlabel("Coefficients")
plt.title("BayesianRidge Coefficients by Wine Type")
plt.legend(handles=legend_elements, title="Wine Type", loc="upper right")
plt.tight_layout()
plt.savefig(PATH+"figures/coefs_br")
# plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=coefs_sgd_white, orient="h", color="skyblue")
sns.boxplot(data=coefs_sgd_red, orient="h", color="tomato",)
plt.axvline(x=0, color="gray", ls="--")
plt.xlabel("Coefficients")
plt.title("SGDRegressor Coefficients by Wine Type")
plt.legend(handles=legend_elements, title="Wine Type", loc="upper right")
plt.tight_layout()
plt.savefig(PATH+"figures/coefs_sgd")
# plt.show()


