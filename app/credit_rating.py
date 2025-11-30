"""
Credit Rating Model
Calculate credit rating, interest rate, and approval result using ML model
"""
from app.model_integration import predict_credit_score, convert_score_to_rating
from datetime import datetime


def calculate_credit_score(
    age: int,
    gender: str,
    ssn: str,
    full_name: str,
    annual_income: float,
    monthly_inhand_salary: float,
    occupation: str,
    month: str = None
) -> dict:
    """
    Calculate credit score, rating, interest rate, and approval result using ML model
    
    Args:
        age: User's age
        gender: User's gender (e.g., "Male", "Female")
        ssn: Social Security Number
        full_name: Full legal name
        annual_income: Annual income
        monthly_inhand_salary: Monthly in-hand salary
        occupation: Occupation
        month: Application month (if None, uses current month)
    
    Returns:
        Dictionary with credit_score, credit_rating, interest_rate, and approved
    """
    # Get current month if not provided
    if month is None:
        month = datetime.now().strftime("%B")  # e.g., "January", "February"
    
    # Prepare user_info dictionary for the model
    user_info = {
        "Age": age,
        "Gender": gender,
        "SSN": ssn,
        "Legal_name": full_name,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Occupation": occupation,
        "Month": month
    }
    
    # Predict credit score using ML model
    try:
        credit_score = predict_credit_score(user_info)
        # Convert score to rating system
        result = convert_score_to_rating(credit_score)
        return result
    except Exception as e:
        # Fallback to default rating if model fails
        print(f"Model prediction error: {e}")
        return {
            "credit_score": 0.0,
            "credit_rating": "C",
            "interest_rate": 0.0,
            "approved": False
        }
