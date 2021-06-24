import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

my_df = pd.read_csv("train.csv")
my_predict_df = pd.read_csv("test.csv")

# Clean the data set.
x_data = my_df[["YearBuilt"]]
y_data = my_df["SalePrice"]
my_predict_df_data = my_predict_df[["YearBuilt"]]

# Build model and predict the price.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
clf = DecisionTreeClassifier()

clf.fit(X=x_train, y=y_train)

graph = clf.predict(my_predict_df_data)
print(clf.score(x_test, y_test))

# Plot the prediction.
plt.plot(graph)
plt.show()
