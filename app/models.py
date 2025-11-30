from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class CreditApplicationCreate(BaseModel):
    ssn: str
    full_name: str
    annual_income: float
    employment_years: float
    credit_history_years: float
    debt_to_income_ratio: float
    loan_amount: float
    loan_term_years: int


class CreditApplicationResponse(BaseModel):
    id: int
    credit_rating: str
    interest_rate: float
    approved: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

