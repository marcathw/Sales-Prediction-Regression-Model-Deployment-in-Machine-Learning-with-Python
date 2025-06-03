# Libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2, mean_absolute_percentage_error as mape
import math
import pickle

class Preprocessor:
    def __init__(self, filetrain, filetest):
        self.filepath_train = filetrain
        self.filepath_test = filetest
    
    def read_data(self):
        self.train = pd.read_csv(self.filepath_train)
        self.test = pd.read_csv(self.filepath_test)

    def feature_engineering(self):
        self.train["Date"] = pd.to_datetime(self.train["Date"])
        self.train["year"] = self.train["Date"].dt.year
        self.train["month"] = self.train["Date"].dt.month
        self.train["day"] = self.train["Date"].dt.day
        self.train = self.train.drop(columns = ["Date"])
        self.test["Date"] = pd.to_datetime(self.test["Date"])
        self.test["year"] = self.test["Date"].dt.year
        self.test["month"] = self.test["Date"].dt.month
        self.test["day"] = self.test["Date"].dt.day
        self.test = self.test.drop(columns = ["Date"])

    def drop_dup(self):
        self.train = self.train.drop_duplicates().reset_index(drop = True)
        self.test = self.test.drop_duplicates().reset_index(drop = True)

    def define_x_y(self):
        x_train = self.train.drop(columns = "number_sold")
        x_test = self.test.drop(columns = "number_sold")
        y_train = self.train["number_sold"]
        y_test = self.test["number_sold"]
        return x_train, x_test, y_train, y_test

class Modeling:
    def __init__(self, x_train, x_test, y_train, y_test, n_estimators = 100, learning_rate = 0.1, max_depth = 3, objective = 'reg:squarederror', random_state = 7):
        self.model = XGBRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, objective = objective, random_state = random_state)

    def train(self):
        self.model.fit(x_train, y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(x_test)
        self.model_mae = mae(y_test, self.y_pred)
        self.model_rmse = math.sqrt(mse(y_test, self.y_pred))
        self.model_r2 = r2(y_test, self.y_pred)
        self.model_mape = mape(y_test, self.y_pred)
        print(f"MAE : {self.model_mae}\nRMSE : {self.model_rmse}\nR2 : {self.model_r2}\nMAPE : {self.model_mape}")

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

# ----------
preprocessor = Preprocessor("train.csv", "test.csv")
preprocessor.read_data()
preprocessor.feature_engineering()
preprocessor.drop_dup()
x_train, x_test, y_train, y_test = preprocessor.define_x_y()

modeling = Modeling(x_train, x_test, y_train, y_test)
modeling.train()
modeling.evaluate()
modeling.model_save("Number Sold.pkl")