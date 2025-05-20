# ML_regression_wine
Objective: using regression models from scikit-learn to predict wine quality, exploring differences in features by wine type, and exploring differences in model coefficients by wine type. 

DATASET: UCI Wine (https://archive.ics.uci.edu/dataset/186/wine+quality)

Key Finds:
- White and Red wines from this dataset differ significantly in many qualities except alcohol content and quality levels. Large differences especially in residual sugar, sulfur dioxide, and acidity.
- White wine quality is largely determined by residual sugar, density, volatile acidity, and alcohol content across multiple regression models.
- Red wine quality is largely determined by volatile acidity, sulphates, and alcohol content across multiple regression models.

Areas for Improvement:
- Regression models could be optimized. Most models have an average mean squared error (MSE) around 50% and an R^2 score around 30%. This is not good, but it's not the point of the project.

