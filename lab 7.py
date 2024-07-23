# decision tree

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree
df = pd.read_csv('/content/weather_forecast.csv')
df.head()
df = pd.get_dummies(df,drop_first = True)
X = df.drop('Play_Yes',axis = 1)
y = df['Play_Yes']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
clf_id3 = DecisionTreeClassifier(criterion = 'entropy',random_state = 42)
clf_id3.fit(X_train,y_train)
plt.figure(figsize = (10,6))
plot_tree(clf_id3,filled = True, feature_names = X.columns, class_names=['No','Yes'])
plt.show()
y_pred_id3 = clf_id3.predict(X_test)
accuracy_id3 = accuracy_score(y_test,y_pred_id3)
report = classification_report(y_test,y_pred_id3)
print("Accuracy: ",accuracy_id3)
print("Classification report ",report)
