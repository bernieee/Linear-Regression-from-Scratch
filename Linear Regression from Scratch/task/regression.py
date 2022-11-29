# write your code here
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit(self, X, y):
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

        if self.fit_intercept:
            self.intercept = coefficients[0]
            self.coefficient = np.array(coefficients[1:])
        else:
            self.intercept = 0.0
            self.coefficient = np.array(coefficients)

    def predict(self, X):
        if self.fit_intercept:

            return X[:, 1:] @ self.coefficient + np.full(X.shape[0], self.intercept)
        else:
            return X @ self.coefficient

    def r2_score(self, y, y_pred):
        return 1 - sum((y - y_pred)**2) / sum((y - np.mean(y))**2)

    def rmse(self, y, y_pred):
        return np.sqrt(sum((y - y_pred)**2) / y.shape[0])


class DataLoader:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.X = ...
        self.y = ...

    def load(self, data_path, sep):
        if self.fit_intercept:
            self.X = pd.read_csv(data_path, sep=sep).iloc[:, 0:-1]
            self.y = pd.read_csv(data_path, sep=sep).iloc[:, -1]

            self.X.insert(loc=0, column='Intercept col', value=np.ones(self.y.values.shape[0]))
        else:
            self.X = pd.read_csv(data_path, sep=sep).iloc[:, 0:-1]
            self.y = pd.read_csv(data_path, sep=sep).iloc[:, -1]

        return self.X, self.y


data_loader = DataLoader(fit_intercept=True)
X_df, y_df = data_loader.load('/home/bernie/Downloads/data_stage4.csv', sep=',')

X = np.array(X_df)
y = np.array(y_df)

sklearn_regression = LinearRegression(fit_intercept=True)
sklearn_regression.fit(X, y)
sklearn_y_pred = sklearn_regression.predict(X)

sklearn_R2 = r2_score(y, sklearn_y_pred)
sklearn_RMSE = np.sqrt(mean_squared_error(y, sklearn_y_pred))

ans_dict = {'Intercept': sklearn_regression.intercept_,
            'Coefficient': sklearn_regression.coef_[1:],
            'R2': sklearn_R2,
            'RMSE': sklearn_R2}

print("Sklearn Linear Regression:\n", ans_dict)

regression = CustomLinearRegression(fit_intercept=True)
regression.fit(X, y)
y_pred = regression.predict(X)

R2 = regression.r2_score(y, y_pred)
RMSE = regression.rmse(y, y_pred)

ans_dict = {'Intercept': regression.intercept,
            'Coefficient': regression.coefficient,
            'R2': R2,
            'RMSE': RMSE}

print("Custom Linear Regression:\n", ans_dict)
