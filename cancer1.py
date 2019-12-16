import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('cancer.csv')
print(dataset)
X = dataset.iloc[:,2:32].values
y = dataset.iloc[:, 1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print X


from sklearn.impute import SimpleImputer

# Fitting missing values
imputer = SimpleImputer(strategy='mean', fill_value='NaN')
imputer = imputer.fit( X[ : , 2:32])
X[ : , 2:32] = imputer.transform(X[ : , 2:32 ])
print(X)
print("\n")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2,
random_state=0)

print("Input for training: \n {}".format(X_train))
print("\n")

print("Output for training: \n {}".format(y_train))
print("\n")

print("Input for testing: \n {}".format(X_test))
print("\n")

print("Output for testing: \n {}".format(y_test))
print("\n")


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


