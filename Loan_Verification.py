## Sample Machine Learning Implementation, simple model
## Generation of a large synthetic dataset (5,000 data points) simulating loan applications. The dataset includes key parameters such as credit score, annual income, employment status, years of employment, down payment, and requested loan amount

# Importing libraries 
import numpy as np
import pandas as pd
## Generation of large synthetic datasets with 5000 data points. 
np.random.seed(42)
n_samples = 5000

data = {
    'credit_score': np.random.randint(550, 850, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'loan_amount': np.random.randint(10000, 300000, n_samples),
    'down_payment': np.random.randint(2000, 80000, n_samples),
    'employment_status': np.random.choice(
        ['employed', 'self_employed', 'unemployed'],
        size=n_samples,
        p=[0.6, 0.25, 0.15]
    ),
    'years_employed': np.random.randint(0, 15, n_samples)
}

df = pd.DataFrame(data)

# View first 10 rows of the generated data
df.head(10)

## Defind loan aproval parameters
def loan_approval(row):
    if (
        row['credit_score'] >= 700 and
        row['income'] >= 30000 and
        row['loan_amount'] <= row['income'] * 4 and ## Multiplying income by 4, a person can safely take a loan up to about 4 times their annual income. Example: If income = $50,000, maximum loan ≈ $200,000.
        row['down_payment'] >= row['loan_amount'] * 0.2 and ##Multiplying loan_amount by 0.2 means the down payment must be at least 20% of the loan amount.  #Example: If loan amount = $100,000,  minimum down payment = $20,000.
        row['employment_status'] != 'unemployed' and
        row['years_employed'] >= 2
    ):
        return 1
    return 0

df['approved'] = df.apply(loan_approval, axis=1)
## Features endcoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Encode categorical feature
encoder = LabelEncoder()
df['employment_status'] = encoder.fit_transform(df['employment_status'])
# Data splitting 
X = df.drop('approved', axis=1)
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% of the data will be use for testing the model while 80% will be use for training 
    random_state=42,
    stratify=y
)
#  Scaling and model development using logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Model predition
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

Model Accuracy: 87.10%

# Testing the model with parameters from a sample customer, for example Mr. Leonel 
## Credit Score: 780

## Annual Income: 65,000

## Loan Amount: 180,000

## Down Payment: 40,000

## Employment Status: Employed

## Years Employed: 4

Leonel = pd.DataFrame({
    'credit_score': [780],
    'income': [65000],
    'loan_amount': [180000],
    'down_payment': [40000],
    'employment_status': encoder.transform(['employed']),
    'years_employed': [4]
})

prediction = pipeline.predict(Leonel)
probability = pipeline.predict_proba(Leonel)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Denied")

print(f"Approval Probability: {probability[0][1]:.2%}")

Loan Denied
Approval Probability: 20.56%

## Testing the model with parameters from a sample customer, for eaxmple Mr. Kenneth 
## Credit Score: 820

## Annual Income: 95,000

## Loan Amount: 200,000

## Down Payment: 60,000

## Employment Status: Employed

## Years Employed: 6

good_customer = pd.DataFrame({
    'credit_score': [820],
    'income': [95000],
    'loan_amount': [200000],
    'down_payment': [60000],
    'employment_status': encoder.transform(['employed']),
    'years_employed': [6]
})

prediction = pipeline.predict(good_customer)
probability = pipeline.predict_proba(good_customer)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Denied")

print(f"Approval Probability: {probability[0][1]:.2%}")

Loan Approved
Approval Probability: 70.93%
    
The trained model was used to evaluate sample individual applicants by transforming input features through the same preprocessing pipeline and producing both binary approval decisions and probability scores.



