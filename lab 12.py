# Multi layer perceptron 

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Data for AND-NOT function
X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([0, 1, 1, 0])

# Data for XOR function
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# AND-NOT MLP
mlp_and_not = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='adam', max_iter=2000, learning_rate_init=0.01)
mlp_and_not.fit(X_and_not, y_and_not)
y_pred_and_not = mlp_and_not.predict(X_and_not)
accuracy_and_not = accuracy_score(y_and_not, y_pred_and_not)
conf_matrix_and_not = confusion_matrix(y_and_not, y_pred_and_not)
print(f"AND-NOT Function Accuracy: {accuracy_and_not * 100}%")
print(f"AND-NOT Predictions: {y_pred_and_not}")
print("AND-NOT Confusion Matrix:")
print(conf_matrix_and_not)

# XOR MLP
mlp_xor = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='adam', max_iter=2000, learning_rate_init=0.01)
mlp_xor.fit(X_xor, y_xor)
y_pred_xor = mlp_xor.predict(X_xor)
accuracy_xor = accuracy_score(y_xor, y_pred_xor)
conf_matrix_xor = confusion_matrix(y_xor, y_pred_xor)
print(f"XOR Function Accuracy: {accuracy_xor * 100}%")
print(f"XOR Predictions: {y_pred_xor}")
print("XOR Confusion Matrix:")
print(conf_matrix_xor)

# Plot confusion matrices
plt.figure(figsize=(12, 5))

# AND-NOT Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_and_not, annot=True, cmap='Blues', fmt='d')
plt.title('AND-NOT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# XOR Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_xor, annot=True, cmap='Blues', fmt='d')
plt.title('XOR Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
