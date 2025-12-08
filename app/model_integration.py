"""
Model Integration Module
Loads the ML model and training data, and provides interface to Credit_Score function
"""
import os
import pandas as pd
import joblib
from datetime import datetime
import sys
import importlib.util
from app.logger_config import get_logger

# Get logger
logger = get_logger()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CSV_PATH = os.path.join(MODEL_DIR, 'cleaned_train_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')

# Load the Credit_score module dynamically
model_file_path = os.path.join(MODEL_DIR, 'Credit_score.py')
spec = importlib.util.spec_from_file_location("Credit_score", model_file_path)
Credit_score_module = importlib.util.module_from_spec(spec)
sys.modules["Credit_score"] = Credit_score_module
spec.loader.exec_module(Credit_score_module)
Credit_Score = Credit_score_module.Credit_Score

# Global variables for loaded data
_df_train = None
_model = None


def load_model_and_data():
    """Load training data and model (call once at startup)"""
    global _df_train, _model
    if _df_train is None or _model is None:
        logger.info("Loading ML model and training data...")
        if not os.path.exists(CSV_PATH):
            logger.error(f"Training data not found at {CSV_PATH}")
            raise FileNotFoundError(f"Training data not found at {CSV_PATH}")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        _df_train = pd.read_csv(CSV_PATH)
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully. Training data shape: {_df_train.shape}")
    return _df_train, _model


def predict_credit_score(user_info: dict) -> float:
    """
    Predict credit score using the ML model
    
    Args:
        user_info: Dictionary with required fields:
            - Age: int
            - Gender: str
            - SSN: str
            - Legal_name: str
            - Annual_Income: float
            - Monthly_Inhand_Salary: float
            - Occupation: str
            - Month: str (January, February, ..., December)
    
    Returns:
        Credit score (float)
    """
    df_train, model = load_model_and_data()
    
    # Ensure all required fields are present
    required_fields = ['Age', 'Gender', 'SSN', 'Legal_name', 'Annual_Income', 
                       'Monthly_Inhand_Salary', 'Occupation', 'Month']
    
    for field in required_fields:
        if field not in user_info:
            raise ValueError(f"Missing required field: {field}")
    
    # Log user input information (without sensitive data)
    logger.info("=" * 80)
    logger.info("Credit Score Prediction Request")
    logger.info(f"User: {user_info.get('Legal_name', 'N/A')}")
    logger.info(f"Age: {user_info.get('Age', 'N/A')}")
    logger.info(f"Gender: {user_info.get('Gender', 'N/A')}")
    logger.info(f"Occupation: {user_info.get('Occupation', 'N/A')}")
    logger.info(f"Annual Income: ${user_info.get('Annual_Income', 'N/A'):,.2f}")
    logger.info(f"Monthly Salary: ${user_info.get('Monthly_Inhand_Salary', 'N/A'):,.2f}")
    logger.info(f"Application Month: {user_info.get('Month', 'N/A')}")
    
    # Call the model function
    logger.info("Calling ML model for prediction...")
    credit_score = Credit_Score(user_info, df_train, model)
    
    logger.info(f"Raw model output: {credit_score}")
    logger.info(f"Model output type: {type(credit_score)}")
    
    # Extract the score value (model.predict returns array)
    if hasattr(credit_score, '__iter__') and not isinstance(credit_score, str):
        credit_score = float(credit_score[0]) if len(credit_score) > 0 else 0.0
    else:
        credit_score = float(credit_score)
    
    logger.info(f"Final Credit Score: {credit_score:.2f}")
    logger.info("=" * 80)
    
    return credit_score


def convert_score_to_rating(credit_score: float) -> dict:
    """
    Convert ML model credit score to rating system (A, B, C)
    
    Args:
        credit_score: Credit score from ML model
    
    Returns:
        Dictionary with credit_rating, interest_rate, and approved
    """
    # Map credit score to rating
    # Adjust thresholds based on your model's score range
    # Assuming score range is similar to typical credit scores (300-850)
    
    logger.info(f"Converting credit score {credit_score:.2f} to rating...")
    
    if credit_score == 0:  # Good credit
        rating = "A"
        interest_rate = 7.0
        approved = True
    elif credit_score == 2:  # Fair credit
        rating = "B"
        interest_rate = 3.5
        approved = True
    else:  # Poor credit
        rating = "C"
        interest_rate = 0.0
        approved = False
    
    logger.info(f"Rating: {rating}, Interest Rate: {interest_rate}%, Approved: {approved}")
    
    result = {
        "credit_score": credit_score,
        "credit_rating": rating,
        "interest_rate": interest_rate,
        "approved": approved
    }
    
    logger.info(f"Final Result: {result}")
    
    return result

