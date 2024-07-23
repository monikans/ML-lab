# Single layer perceptron

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # A AND B

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # A OR B

perceptron_and = Perceptron(max_iter=1000, random_state=0)
perceptron_and.fit(X_and, y_and)
y_pred_and = perceptron_and.predict(X_and)
accuracy_and = accuracy_score(y_and, y_pred_and)
conf_matrix_and = confusion_matrix(y_and, y_pred_and)

perceptron_or = Perceptron(max_iter=1000, random_state=0)
perceptron_or.fit(X_or, y_or)
y_pred_or = perceptron_or.predict(X_or)
accuracy_or = accuracy_score(y_or, y_pred_or)
conf_matrix_or = confusion_matrix(y_or, y_pred_or)

print(f"AND Function Accuracy: {accuracy_and * 100}%")
print(f"AND Predictions: {y_pred_and}")
print("AND Confusion Matrix:")
print(conf_matrix_and)

print(f"\nOR Function Accuracy: {accuracy_or * 100}%")
print(f"OR Predictions: {y_pred_or}")
print("OR Confusion Matrix:")
print(conf_matrix_or)

plt.figure(figsize=(12,5))

sns.heatmap(conf_matrix_and, annot=True,cmap='Blues')
plt.title('AND Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(12,5))
sns.heatmap(conf_matrix_or, annot=True, cmap='Blues')
plt.title('OR Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
