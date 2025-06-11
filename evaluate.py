import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    # Looking at mean squared errored and R^2
    mse = np.round(mean_squared_error(y_test, y_pred), decimals=5)
    r2 = np.round(r2_score(y_test, y_pred), decimals=5)
    
    return mse, r2

def cv_coefs(model, x, y, features, seed):
    
    cv = cross_validate(
        model,
        x,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed),
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