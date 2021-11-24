# Random forest classification
# https://archive.ics.uci.edu/ml/datasets/wine
# Using chemical analysis determine the origin of wines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# import data
df = pd.read_csv('Wine_Quality_Data.csv')
# define target feature
target = 'color'
# remove missing values
df.dropna(inplace=True)
# labels are already encoded, no further data preprocessing necessary here for decision tree
# define x and y
X, y = df.drop(columns=target), df[target]
# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # shuffle True default
# Initialize the random forest estimator
RF = RandomForestClassifier(oob_score=True,
                            bootstrap=True,
                            random_state=42,
                            warm_start=True,
                            n_jobs=-1)
# initialize empy out-of-bag error list
oob_error_list = []
# Iterate through different number of trees
n_trees_param = [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]
for n_trees in n_trees_param:
    # Use this to set the number of trees
    RF.set_params(n_estimators=n_trees)
    # Fit the model
    RF.fit(X_train, y_train)
    # Get the oob error
    oob_error = 1 - RF.oob_score_
    # Store it
    oob_error_list.append(oob_error)
# store into pandas serie
oob_error = pd.Series(oob_error_list, index =n_trees_param)
# retrieve best param n tree
best_n_tree = oob_error.idxmin(axis=1)
print("Min error for n tree : ", best_n_tree)
# Random forest with best n_tree
model = RF.set_params(n_estimators=best_n_tree)
# run the model on X_test
y_pred = model.predict(X_test)
# evaluate results
cr = classification_report(y_test, y_pred)
print(cr)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap=plt.cm.Blues)
plt.show()