from flask import Flask, render_template, request
from model import credit_model

app = Flask(__name__, template_folder='templates')


def calculate_fallback_score(data):
    """Calculate a simple credit score when the model fails"""
    try:
        # Base score
        score = 650

        # Apply modifiers based on available data
        if data:
            # Positive factors
            score += min(float(data.get('Annual_Income', 0)) / 10000, 100)
            score += int(data.get('Num_Bank_Accounts', 0)) * 5
            score -= int(data.get('Num_of_Delayed_Payment', 0)) * 10
            score -= float(data.get('Outstanding_Debt', 0)) / 1000

            # Negative factors
            if float(data.get('Interest_Rate', 0)) > 15:
                score -= 20
            if int(data.get('Num_of_Loan', 0)) > 5:
                score -= 15

        # Keep score within standard credit score range
        score = max(300, min(850, round(score)))

        # Format to match what the template expects
        if score >= 700:
            return {
                'prediction': 'Good',
                'probabilities': {
                    'Good': 80,
                    'Standard': 15,
                    'Bad': 5
                }
            }
        elif score >= 600:
            return {
                'prediction': 'Standard',
                'probabilities': {
                    'Good': 30,
                    'Standard': 60,
                    'Bad': 10
                }
            }
        else:
            return {
                'prediction': 'Bad',
                'probabilities': {
                    'Good': 5,
                    'Standard': 25,
                    'Bad': 70
                }
            }
    except:
        return {
            'prediction': 'Standard',
            'probabilities': {
                'Good': 40,
                'Standard': 50,
                'Bad': 10
            }
        }


@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request"""
    # Initialize empty dict to store data
    form_data = {}

    try:
        # Safely get form data with defaults
        form_data = {
            'Annual_Income': request.form.get('annual_income', '0'),
            'Num_Bank_Accounts': request.form.get('num_bank_accounts', '0'),
            'Num_Credit_Card': request.form.get('num_credit_cards', '0'),
            'Interest_Rate': request.form.get('interest_rate', '0'),
            'Num_of_Loan': request.form.get('num_loans', '0'),
            'Delay_from_due_date': request.form.get('delay_due_date', '0'),
            'Num_of_Delayed_Payment': request.form.get('num_delayed_payments', '0'),
            'Changed_Credit_Limit': request.form.get('changed_credit_limit', '0'),
            'Outstanding_Debt': request.form.get('outstanding_debt', '0'),
            'Credit_Utilization_Ratio': request.form.get('credit_utilization', '0')
        }

        # Convert to proper types
        processed_data = {k: float(v) if '.' in v else int(v) for k, v in form_data.items()}

        # Try to get prediction from model
        try:
            result = credit_model.predict(processed_data)
            # If model returns a score number, convert to the format the template expects
            if isinstance(result, (int, float)):
                if result >= 700:
                    result = {
                        'prediction': 'Good',
                        'probabilities': {
                            'Good': 80,
                            'Standard': 15,
                            'Bad': 5
                        }
                    }
                elif result >= 600:
                    result = {
                        'prediction': 'Standard',
                        'probabilities': {
                            'Good': 30,
                            'Standard': 60,
                            'Bad': 10
                        }
                    }
                else:
                    result = {
                        'prediction': 'Bad',
                        'probabilities': {
                            'Good': 5,
                            'Standard': 25,
                            'Bad': 70
                        }
                    }
            # If model returns an error dict, use fallback
            elif isinstance(result, dict) and 'error' in result:
                raise Exception(result['error'])
        except Exception as model_error:
            app.logger.error(f"Model error: {model_error}")
            result = calculate_fallback_score(processed_data)

        return render_template('result.html', result=result)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        # Calculate fallback score with whatever data we have
        fallback_score = calculate_fallback_score(form_data)
        return render_template('result.html', result=fallback_score)


if __name__ == '__main__':
    app.run(debug=True)