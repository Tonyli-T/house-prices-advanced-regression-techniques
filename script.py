# Author Yizhou Li, e-mail: lyzpp2000@163.com

# Housing Price Prediction Assignment

# Problem Statement:

# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling
# or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences
# price negotiations than the number of bedrooms or a white-picket fence.
#
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition
# challenges you to predict the final price of each home.

# Steps:

# Importing modules

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Helper functions

def load_data():
    train_df = pd.read_csv("train.csv", index_col='Id')
    test_df = pd.read_csv("test.csv", index_col='Id')
    df = pd.concat([train_df, test_df])
    # print(df.head())
    # print(df.info())
    print(df.groupby("YearBuilt"))
    # price_metadata_group = price_metadata.groupby("Null_Percentage").count().reset_index()


# 1. Data loading and exploring     --TODO

load_data()

# 2. Data cleaning

# # 3. Data visualization
#
# # 4. Data preparation
#
# # 5. Spiting and scaling the data
#
# # 6. Models building and evaluation
# x_data = my_df[["YearBuilt"]]
# y_data = my_df["SalePrice"]
# my_predict_df_data = my_predict_df[["YearBuilt"]]
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
# clf = DecisionTreeClassifier()
#
# clf.fit(X=x_train, y=y_train)
#
# graph = clf.predict(my_predict_df_data)
# print(clf.score(x_test, y_test))
#
# # 7. Using the best model and input data to predict the housing price
