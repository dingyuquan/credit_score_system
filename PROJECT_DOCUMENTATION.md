# Credit Application System - Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Code Structure](#code-structure)
8. [Machine Learning Integration](#machine-learning-integration)
9. [Database Schema](#database-schema)
10. [Logging System](#logging-system)
11. [Future Improvements](#future-improvements)

---

## Project Overview

The **Credit Application System** is a web-based application that allows users to apply for credit online. The system uses machine learning to evaluate credit applications and automatically determines credit ratings, interest rates, and approval status based on user-provided financial and personal information.

### Key Highlights
- **Automated Credit Assessment**: ML-powered credit scoring replaces manual evaluation
- **User-Friendly Interface**: Modern, responsive web design with intuitive forms
- **Secure Authentication**: Session-based user authentication with password encryption
- **Comprehensive Logging**: Detailed logging of all predictions and system events
- **Real-time Results**: Instant credit rating and approval decisions

---

## Features

### User Features
1. **User Registration & Authentication**
   - Secure user account creation
   - Login/logout functionality
   - Session-based authentication

2. **Credit Application Submission**
   - Comprehensive application form collecting:
     - Personal information (SSN, name, age, gender)
     - Financial information (annual income, monthly salary)
     - Employment details (occupation, application month)
   - Real-time form validation

3. **Application Results**
   - Instant credit rating (A, B, or C)
   - Interest rate determination based on rating
   - Approval/rejection status
   - Detailed result display

4. **Application History**
   - Dashboard showing all submitted applications
   - View detailed information for each application
   - Status tracking (Approved/Rejected)

### System Features
1. **Machine Learning Integration**
   - Pre-trained credit scoring model
   - Automatic feature preprocessing
   - Score-to-rating conversion

2. **Comprehensive Logging**
   - All predictions logged to daily log files
   - User input tracking (non-sensitive data)
   - Model output recording
   - System event logging

3. **Error Handling**
   - Graceful fallback if model fails to load
   - Input validation
   - User-friendly error messages

---

## Technology Stack

### Backend Technologies
- **FastAPI** (v0.104.1): Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **SQLAlchemy** (v2.0.23): ORM for database operations
- **SQLite**: Lightweight relational database
- **Starlette Sessions**: Session middleware for authentication

### Frontend Technologies
- **Jinja2** (v3.1.2): Server-side templating engine
- **HTML5 & CSS3**: Modern web standards
- **Custom CSS**: Professional, responsive design
- **JavaScript**: Client-side form enhancements

### Machine Learning & Data Science
- **pandas** (v2.1.4): Data manipulation and analysis
- **scikit-learn** (v1.3.2): Machine learning preprocessing
  - OneHotEncoder for categorical features
  - StandardScaler for numerical features
- **xgboost** (v2.0.3): Gradient boosting model (loaded via joblib)
- **joblib** (v1.3.2): Model serialization/deserialization

### Security & Authentication
- **passlib[bcrypt]**: Password hashing and verification
- **python-jose**: JWT token support (for future API expansion)

### Development Tools
- **Python 3.9+**: Programming language
- **pydantic**: Data validation
- **Python logging**: Comprehensive logging system

---

## System Architecture

### Architecture Overview

The system follows a **layered architecture** pattern:

```
┌─────────────────────────────────────────┐
│      Presentation Layer (Templates)     │
│  Jinja2 Templates + Static CSS/JS      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Application Layer (FastAPI)       │
│  Routes, Request Handling, Validation   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Business Logic Layer               │
│  Credit Rating Calculation              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      ML Integration Layer               │
│  Model Loading, Prediction, Conversion  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Data Access Layer                  │
│  SQLAlchemy ORM, Database Operations    │
└─────────────────────────────────────────┘
```

### Component Interaction Flow

1. **User Request Flow**:
   - User submits application form → FastAPI route receives data
   - Route validates input → Calls credit rating function
   - Rating function prepares data → Calls ML model integration
   - ML model processes input → Returns credit score
   - Score converted to rating → Result saved to database
   - Response rendered → User sees result page

2. **Model Prediction Flow**:
   - User input collected → Formatted as dictionary
   - Data preprocessing (one-hot encoding, scaling) → Feature alignment
   - Model prediction → Raw credit score
   - Score conversion → Rating (A/B/C) + Interest rate
   - Result logging → Written to log file

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone or Download Project
```bash
# Navigate to project directory
cd Intro2Prog_Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model Files
Ensure the following files exist in the `model/` directory:
- `cleaned_train_data.csv` - Training dataset
- `best_model.joblib` - Pre-trained ML model
- `Credit_score.py` - Model prediction function

### Step 4: Run the Application
```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://localhost:8000
```

---

## Usage Guide

### For End Users

#### 1. Register an Account
- Click "Register" on the homepage
- Fill in username, email, and password
- Submit the form

#### 2. Login
- Click "Login" and enter your credentials
- You'll be redirected to the dashboard

#### 3. Submit a Credit Application
- Click "Get Started" or "Submit New Application"
- Fill in all required fields:
  - **Personal Information**: SSN, Full Name, Age, Gender
  - **Financial Information**: Annual Income, Monthly In-Hand Salary
  - **Employment**: Occupation, Application Month
- Click "Submit Application"

#### 4. View Results
- Results are displayed immediately after submission
- View includes:
  - Credit Rating (A, B, or C)
  - Interest Rate (if approved)
  - Approval Status

#### 5. View Application History
- Access your dashboard to see all applications
- Click "View Details" to see full application information

### For Administrators/Developers

#### Viewing Logs
Logs are stored in the `logs/` directory:
- Format: `credit_app_YYYYMMDD.log`
- Contains:
  - System startup messages
  - Model loading status
  - All prediction requests with inputs/outputs
  - Application processing records

#### Database Access
The SQLite database is stored as `credit_app.db` in the project root. You can access it using:
- SQLite command-line tool
- Database browser applications
- Python scripts using SQLAlchemy

---

## Code Structure

### Project Directory Structure
```
Intro2Prog_Project/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application & routes
│   ├── database.py              # Database models & configuration
│   ├── models.py                # Pydantic schemas
│   ├── auth.py                  # Authentication functions
│   ├── credit_rating.py          # Credit rating business logic
│   ├── model_integration.py     # ML model integration layer
│   ├── logger_config.py         # Logging configuration
│   └── middleware.py            # Middleware settings
│
├── model/                        # Machine learning model
│   ├── Credit_score.py          # Core ML prediction function
│   ├── cleaned_train_data.csv   # Training dataset
│   └── best_model.joblib        # Serialized trained model
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                # Base template with navigation
│   ├── index.html               # Homepage
│   ├── login.html               # Login page
│   ├── register.html            # Registration page
│   ├── dashboard.html           # User dashboard
│   ├── apply.html               # Application form
│   └── result.html              # Result display page
│
├── static/                       # Static assets
│   └── style.css                # Custom CSS styles
│
├── logs/                         # Application logs (auto-generated)
│   └── credit_app_YYYYMMDD.log  # Daily log files
│
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
├── README.md                     # Basic project readme
└── credit_app.db                 # SQLite database (auto-generated)
```

### Key Files Description

#### `app/main.py`
- **Purpose**: Main FastAPI application
- **Key Functions**:
  - Application initialization
  - Route definitions (home, login, register, dashboard, apply)
  - Request handling and form processing
  - Session management

#### `app/database.py`
- **Purpose**: Database configuration and models
- **Key Components**:
  - `User` model: User account information
  - `CreditApplication` model: Credit application records
  - Database initialization functions

#### `app/credit_rating.py`
- **Purpose**: Business logic for credit rating
- **Key Function**: `calculate_credit_score()`
  - Accepts user input parameters
  - Calls ML model integration
  - Returns rating, interest rate, and approval status

#### `app/model_integration.py`
- **Purpose**: ML model integration layer
- **Key Functions**:
  - `load_model_and_data()`: Loads model and training data on startup
  - `predict_credit_score()`: Calls ML model for prediction
  - `convert_score_to_rating()`: Converts numeric score to A/B/C rating

#### `model/Credit_score.py`
- **Purpose**: Core ML prediction function
- **Key Function**: `Credit_Score(user_info, df_train, model)`
  - Preprocesses user input (one-hot encoding, scaling)
  - Aligns features with training data
  - Returns predicted credit score

---

## Machine Learning Integration

### Model Architecture

The system uses a **pre-trained machine learning model** (likely XGBoost) that has been trained on historical credit data. The model predicts credit scores based on multiple features.

### Feature Engineering

#### Input Features Required:
1. **Age** (int): Applicant's age
2. **Gender** (str): Male, Female, or Other
3. **SSN** (str): Social Security Number
4. **Legal_name** (str): Full legal name
5. **Annual_Income** (float): Annual income in dollars
6. **Monthly_Inhand_Salary** (float): Monthly take-home salary
7. **Occupation** (str): Job occupation
8. **Month** (str): Application month (January-December)

#### Preprocessing Steps:
1. **Categorical Encoding**: One-hot encoding for categorical features (Gender, Occupation, Month)
2. **Numerical Scaling**: StandardScaler normalization for numerical features
3. **Feature Alignment**: Missing features filled with training data modes/means
4. **Feature Selection**: Uses optimal feature subset for prediction

### Credit Rating System

The ML model outputs a numeric credit score, which is then converted to a rating:

| Credit Score Range | Rating | Interest Rate | Approved |
|-------------------|--------|---------------|----------|
| ≥ 700             | A      | 3.5%          | Yes      |
| 600 - 699         | B      | 7.0%          | Yes      |
| < 600             | C      | 0.0%          | No       |

**Note**: These thresholds can be adjusted in `app/model_integration.py` based on your model's actual score distribution.

### Model Loading Strategy

- **Lazy Loading**: Model and training data loaded once at application startup
- **Caching**: Loaded data cached in memory for fast subsequent predictions
- **Error Handling**: Graceful fallback if model fails to load

---

## Database Schema

### Users Table
| Column          | Type    | Description                    |
|----------------|---------|--------------------------------|
| id              | Integer | Primary key                    |
| username        | String  | Unique username                |
| email           | String  | Unique email address           |
| hashed_password | String  | Bcrypt hashed password         |
| created_at      | DateTime| Account creation timestamp     |

### Credit Applications Table
| Column                | Type    | Description                          |
|----------------------|---------|--------------------------------------|
| id                   | Integer | Primary key                          |
| user_id              | Integer | Foreign key to users table           |
| ssn                  | String  | Social Security Number               |
| full_name            | String  | Applicant's full name                |
| age                  | Integer | Applicant's age                      |
| gender               | String  | Gender (Male/Female/Other)           |
| annual_income        | Float   | Annual income                        |
| monthly_inhand_salary| Float   | Monthly take-home salary             |
| occupation           | String  | Job occupation                       |
| month                | String  | Application month                    |
| credit_score         | Float   | ML model predicted score             |
| credit_rating        | String  | Rating (A/B/C)                       |
| interest_rate        | Float   | Interest rate percentage             |
| approved             | Boolean | Approval status                      |
| created_at           | DateTime| Application submission timestamp     |

---

## Logging System

### Log Configuration

The system uses Python's `logging` module with custom configuration:

- **Log Location**: `logs/credit_app_YYYYMMDD.log`
- **Log Format**: Timestamp, Logger Name, Level, Message
- **Output**: Both file and console
- **Rotation**: Daily log files (one file per day)

### Logged Information

#### System Events:
- Application startup
- Database initialization
- Model loading success/failure

#### Prediction Events:
- User input summary (non-sensitive data)
- Model raw output
- Credit score conversion
- Final rating and approval decision

#### Application Events:
- User registration
- User login
- Application submission
- Application processing results

### Example Log Entry
```
2024-12-01 10:30:45 - credit_app - INFO - ================================================================================
2024-12-01 10:30:45 - credit_app - INFO - Credit Score Prediction Request
2024-12-01 10:30:45 - credit_app - INFO - User: John Doe
2024-12-01 10:30:45 - credit_app - INFO - Age: 35
2024-12-01 10:30:45 - credit_app - INFO - Gender: Male
2024-12-01 10:30:45 - credit_app - INFO - Occupation: Engineer
2024-12-01 10:30:45 - credit_app - INFO - Annual Income: $75,000.00
2024-12-01 10:30:45 - credit_app - INFO - Monthly Salary: $6,250.00
2024-12-01 10:30:45 - credit_app - INFO - Application Month: January
2024-12-01 10:30:45 - credit_app - INFO - Calling ML model for prediction...
2024-12-01 10:30:45 - credit_app - INFO - Raw model output: [650.5]
2024-12-01 10:30:45 - credit_app - INFO - Final Credit Score: 650.50
2024-12-01 10:30:45 - credit_app - INFO - Rating: B, Interest Rate: 7.0%, Approved: True
2024-12-01 10:30:45 - credit_app - INFO - ================================================================================
```

---

## Security Considerations

### Current Security Measures
1. **Password Hashing**: Bcrypt password encryption
2. **Session Management**: Secure session-based authentication
3. **Input Validation**: Form validation on both client and server side
4. **SQL Injection Protection**: SQLAlchemy ORM prevents SQL injection

### Recommendations for Production
1. **HTTPS**: Use SSL/TLS certificates for encrypted connections
2. **Environment Variables**: Move secrets to environment variables
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **CSRF Protection**: Add CSRF tokens to forms
5. **Database Encryption**: Encrypt sensitive data at rest
6. **Audit Logging**: Enhanced logging for security events

---

## Future Improvements

### Potential Enhancements

1. **API Endpoints**
   - RESTful API for mobile applications
   - API documentation with Swagger/OpenAPI

2. **Advanced Features**
   - Email notifications for application status
   - PDF report generation
   - Application status tracking
   - Multi-factor authentication

3. **Model Improvements**
   - Model retraining pipeline
   - A/B testing for different models
   - Model versioning
   - Feature importance visualization

4. **User Experience**
   - Application progress saving (draft applications)
   - Application editing before submission
   - Advanced filtering and search in dashboard
   - Export application history

5. **Administrative Features**
   - Admin dashboard
   - User management
   - Application review interface
   - Analytics and reporting

6. **Performance**
   - Database connection pooling
   - Caching layer (Redis)
   - Async file operations
   - Load balancing for high traffic

7. **Testing**
   - Unit tests for business logic
   - Integration tests for API endpoints
   - End-to-end testing
   - Model prediction accuracy tests

---

## Troubleshooting

### Common Issues

#### Issue: Model fails to load
**Solution**: 
- Verify `model/best_model.joblib` exists
- Check file permissions
- Ensure xgboost is installed: `pip install xgboost`

#### Issue: Database errors
**Solution**:
- Delete `credit_app.db` and restart (will recreate)
- Check SQLite is available
- Verify database file permissions

#### Issue: Import errors
**Solution**:
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.9+)
- Verify virtual environment is activated

#### Issue: Logs not being created
**Solution**:
- Check `logs/` directory exists and is writable
- Verify logger configuration in `app/logger_config.py`
- Check application has write permissions

---

## Contact & Support

For questions, issues, or contributions, please refer to the project repository or contact the development team.

---

## License

[Specify your license here]

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Project**: Credit Application System

