from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from datetime import datetime

# Database configuration
DATABASE_URL = "sqlite+aiosqlite:///./credit_app.db"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Credit application model
class CreditApplication(Base):
    __tablename__ = "credit_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ssn = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    
    # Model input fields
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    annual_income = Column(Float, nullable=False)
    monthly_inhand_salary = Column(Float, nullable=False)
    occupation = Column(String, nullable=False)
    month = Column(String, nullable=False)
    
    # Legacy fields (kept for backward compatibility)
    employment_years = Column(Float, nullable=True)
    credit_history_years = Column(Float, nullable=True)
    debt_to_income_ratio = Column(Float, nullable=True)
    loan_amount = Column(Float, nullable=True)
    loan_term_years = Column(Integer, nullable=True)
    
    # Rating results
    credit_score = Column(Float, nullable=True)  # ML model output
    credit_rating = Column(String, nullable=True)
    interest_rate = Column(Float, nullable=True)
    approved = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
