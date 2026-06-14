import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Load dataset (assumes DigitalAd_dataset.csv is in the same directory)
dataset = pd.read_csv('DigitalAd_dataset.csv')

# Split features and target
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Train‑test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

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

# Evaluate on test set (optional)
accuracy = best_model.score(X_test_scaled, y_test)
print(f"Best model test accuracy: {accuracy * 100:.2f}%")

# Save the scaler and the best model for later inference
dump(scaler, 'scaler.joblib')
dump(best_model, 'logreg_model.joblib')
