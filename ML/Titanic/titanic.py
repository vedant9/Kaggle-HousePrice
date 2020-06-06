import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#train_data.info()
y = train_data["Survived"]
train_data["Age"].fillna((train_data["Age"].mean()), inplace = True)
train_data["Embarked"].fillna(method = 'ffill', inplace = True)
test_data["Age"].fillna((train_data["Age"].mean()), inplace = True)
test_data["Fare"].fillna(method = 'ffill', inplace = True)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch","Ticket", "Fare", "Embarked"]
number = preprocessing.LabelEncoder()
train_data["Sex"] = number.fit_transform(train_data["Sex"])
test_data["Sex"] = number.fit_transform(test_data["Sex"])
train_data["Ticket"] = number.fit_transform(train_data["Ticket"])
test_data["Ticket"] = number.fit_transform(test_data["Ticket"])
train_data["Embarked"] = number.fit_transform(train_data["Embarked"])
test_data["Embarked"] = number.fit_transform(test_data["Embarked"])
#train_data.info()
#test_data.info()
X = train_data[features]
X_test = test_data[features]
X_reg_new=SelectKBest(score_func=f_regression, k='all').fit(X,y)
#print(X_reg_new.scores_)
#print(X_reg_new)
#HEATMAP ######################################################################
corrmat = train_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (9,9))
g=sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)