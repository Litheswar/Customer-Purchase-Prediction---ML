# -*- coding: utf-8 -*-
\"\"\"customer_purchase_prediction.py

This script trains a Logistic Regression model on the DigitalAd_dataset.csv,
performs hyper-parameter tuning with GridSearchCV, saves the best model and
the StandardScaler, and visualises the decision boundary together with the
actual data points.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('DigitalAd_dataset.csv')

# Split features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grid for LogisticRegression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [200]
}

logreg = LogisticRegression(random_state=0)
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# Evaluate on test set
accuracy = best_model.score(X_test_scaled, y_test)
print(f"Best model test accuracy: {accuracy * 100:.2f}%")

# Save the scaler and the best model for later inference
dump(scaler, 'scaler.joblib')
dump(best_model, 'logreg_model.joblib')

# -------------------------------------------------------
# Visualization: decision boundary with actual points
# -------------------------------------------------------

def plot_decision_boundary(X, y, model, scaler):
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1000))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu, levels=[-0.1, 0.5, 1.1])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.title('Decision Boundary (Blue = Won\'t Buy, Red = Will Buy)')
    plt.show()

# Plot using the training data
plot_decision_boundary(X_train, y_train, best_model, scaler)
