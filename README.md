# Credit Application System

A web-based credit application system developed with FastAPI and Jinja2.

## Features

- User registration and login
- Submit credit applications (including SSN, name, annual income, etc.)
- Intelligent credit rating based on user data
- Different ratings correspond to different interest rates and application results
- View application history

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Jinja2 template engine
- **Database**: SQLite (using SQLAlchemy ORM)
- **Authentication**: Session-based authentication
- **Password Encryption**: bcrypt

## Installation and Running

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python run.py
```

Or run directly with uvicorn:

```bash
uvicorn app.main:app --reload
```

### 3. Access the Application

Open your browser and visit: http://localhost:8000

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI main application
│   ├── database.py          # Database models and configuration
│   ├── models.py            # Pydantic models
│   ├── auth.py              # Authentication functions
│   ├── credit_rating.py     # Credit rating algorithm
│   └── middleware.py        # Middleware configuration
├── templates/               # Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── apply.html
│   └── result.html
├── static/                  # Static files
│   └── style.css
├── requirements.txt
├── run.py                   # Startup script
└── README.md
```

## Usage Instructions

1. **Register Account**: Visit `/register` to create a new account
2. **Login**: Use your registered username and password to log in
3. **Submit Application**: Click "Submit New Application" on the dashboard and fill in the required information
4. **View Results**: Results are displayed immediately after submission, including rating, interest rate, and approval status
5. **View History**: Check all historical application records on the dashboard

## Notes

- Please change the secret keys in `app/middleware.py` and `app/auth.py` for production environments
- The database file `credit_app.db` will be automatically created on first run
- If user data is missing, the system will use training data means to fill in the values

## Development

This project is developed with Python 3.8+.
