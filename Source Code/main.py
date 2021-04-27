# importing packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model  import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 

# Loading Datasets
bccd = pd.read_csv('Datasets/BCCD.csv')
wbcd = pd.read_csv('Datasets/WBCD.csv')

# Machine Learning Model for BCCD dataset.

## Root map
## 1 --> Healthy
## 2 --> Patient

# To check the head
print(bccd.head())

print()

# To check the value counts of the classification
print('Value counts of the classification')
print(bccd['Classification'].value_counts())

# To check correlation map
cor = bccd.corr()

# Plotting correlation map
plt.figure(figsize=(15,8))
sns.heatmap(cor, annot = True)
plt.show()

# Perform Machine Learning Stuff

# Train Test and Split the dataset
X = bccd.drop('Classification',axis=1)
y = bccd['Classification']

X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.3,random_state=101)

# Logistic Regression
logisticreg = LogisticRegression(max_iter=10000)
# Training the model
logisticreg.fit(X_train,y_train)

# Predicting the model
predict1 = logisticreg.predict(X_test)
plot_roc_curve(logisticreg,X_test, y_test, ax=plt.gca())

# Random Forest
rm = RandomForestClassifier(n_estimators=200)
# Training the model
rm.fit(X_train,y_train)
# Predicting the model
predict3 = rm.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
# Training the model
dt.fit(X_train,y_train)
# Predicting the model
predict4 = dt.predict(X_test)


# Printing the classification report 
print('Classification Report for Logistic Regression:' + '\n\n' + classification_report(y_test,predict1))
print('Classification Report for Random Forest:' + '\n\n' + classification_report(y_test,predict3))
print('Classification Report for Decision Tree:' + '\n\n' + classification_report(y_test,predict4))


classifiers = [logisticreg,dt,rm]
ax = plt.gca()
for i in classifiers:
    plot_roc_curve(i, X_test, y_test, ax=ax)
plt.show()