# KNN 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Glass dataset
df = pd.read_csv('/content/glass.csv')

# Features and target variable
X  = df.drop(columns=['Type']) #capital x and small y
y = df['Type']
df.head()

# Train and evaluate KNN classifier with specified metric
def knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    metrics = ['euclidean', 'manhattan']
    results = {}

    for metric in metrics:
        model = KNeighborsClassifier(n_neighbors=3, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[metric] = accuracy
        print(f"{metric.capitalize()} distance: Accuracy = {accuracy:.4f}")

    return results

# Get the accuracy results
results = knn(X, y)

# Visualization of accuracy results
def plot_results(results):
    metrics = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12,8))
    plt.bar(metrics, accuracies, color=['blue', 'orange'])
    plt.xlabel('Distance Metric')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of KNN Classifier with Different Distance Metrics')
    plt.show()

plot_results(results)
