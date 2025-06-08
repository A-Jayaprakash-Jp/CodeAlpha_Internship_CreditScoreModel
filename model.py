# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np


class CreditScoreModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
            'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
            'Outstanding_Debt', 'Credit_Utilization_Ratio'
        ]
        self.model_path = 'credit_score_model.pkl'
        self.scaler_path = 'scaler.pkl'

    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Remove underscores from numeric columns and convert to float
        numeric_cols = [
            'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Delay_from_due_date', 'Num_of_Delayed_Payment',
            'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Outstanding_Debt', 'Credit_Utilization_Ratio',
            'Total_EMI_per_month', 'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        for col in numeric_cols:
            if col in df.columns:
                # Remove underscores and convert to numeric
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('_', ''), errors='coerce')

        # Drop rows with missing values in our feature columns
        df = df.dropna(subset=self.features)

        return df

    def load_data(self):
        """Load and preprocess the data"""
        df = pd.read_csv('test.csv')
        df = self.clean_data(df)

        # Convert Credit_Mix to numerical values
        df['Credit_Mix'] = df['Credit_Mix'].map({'Good': 2, 'Standard': 1, 'Bad': 0, '_': 1})
        df['Credit_Mix'] = df['Credit_Mix'].fillna(1)

        # Create target variable (simplified credit score categories)
        df['Credit_Score'] = pd.cut(df['Credit_Utilization_Ratio'],
                                    bins=[0, 20, 40, 100],
                                    labels=['Good', 'Standard', 'Bad'])

        # Ensure we have enough samples in each category
        if len(df['Credit_Score'].value_counts()) < 3:
            raise ValueError("Not enough samples in all credit score categories")

        return df

    def train_model(self):
        """Train the credit score prediction model"""
        df = self.load_data()

        X = df[self.features]
        y = df['Credit_Score']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save model and scaler
        self.save_model()

    def save_model(self):
        """Save the trained model and scaler"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_saved_model(self):
        """Load the saved model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict(self, input_data):
        """Make a prediction using the trained model"""
        if not self.model or not self.scaler:
            if not self.load_saved_model():
                raise Exception("Model not trained or saved model not found")

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale features
        scaled_data = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(scaled_data)[0]

        # Get prediction probabilities
        probabilities = self.model.predict_proba(scaled_data)[0]

        # Create result dictionary
        result = {
            'prediction': prediction,
            'probabilities': {
                'Good': round(probabilities[0] * 100, 2),
                'Standard': round(probabilities[1] * 100, 2),
                'Bad': round(probabilities[2] * 100, 2)
            }
        }

        return result


# Initialize the model when this module is imported
credit_model = CreditScoreModel()
if not credit_model.load_saved_model():
    print("Training model for the first time...")
    try:
        credit_model.train_model()
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        # Create dummy model if training fails
        credit_model.model = RandomForestClassifier(n_estimators=10, random_state=42)
        credit_model.scaler = StandardScaler()
        # Create dummy training data
        X_dummy = np.random.rand(10, len(credit_model.features))
        y_dummy = np.random.choice(['Good', 'Standard', 'Bad'], size=10)
        credit_model.scaler.fit(X_dummy)
        credit_model.model.fit(credit_model.scaler.transform(X_dummy), y_dummy)
        print("Created a dummy model for demonstration purposes")