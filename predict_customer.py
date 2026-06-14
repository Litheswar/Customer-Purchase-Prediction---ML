import numpy as np
from joblib import load

def predict_customer():
    """Load the persisted scaler and logistic regression model,
    prompt the user for Age and Salary, and output the purchase prediction.
    """
    # Load the artifacts saved during training (they reside in the same directory)
    scaler = load('scaler.joblib')
    model = load('logreg_model.joblib')

    try:
        age = float(input('Enter Customer Age: '))
        salary = float(input('Enter Customer Salary: '))
    except ValueError:
        print('Invalid input – please enter numeric values.')
        return

    # Prepare the feature vector and apply the same scaling used during training
    X_new = np.array([[age, salary]])
    X_scaled = scaler.transform(X_new)
    pred = model.predict(X_scaled)[0]

    if pred == 1:
        print('Customer will Buy')
    else:
        print("Customer won't Buy")

if __name__ == '__main__':
    predict_customer()
