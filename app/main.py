from fastapi import FastAPI, Request, Depends, HTTPException, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import timedelta
import os

from app.database import get_db, init_db, User, CreditApplication
from app.models import UserCreate, CreditApplicationCreate
from app.auth import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_user_by_username,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.credit_rating import calculate_credit_score
from app.middleware import SECRET_KEY

app = FastAPI(title="Credit Application System")

# Add Session middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize database and load ML model
@app.on_event("startup")
async def startup_event():
    # Setup logging
    from app.logger_config import setup_logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Credit Application System Starting...")
    logger.info("=" * 80)
    
    await init_db()
    logger.info("Database initialized")
    
    # Pre-load ML model and training data
    try:
        from app.model_integration import load_model_and_data
        load_model_and_data()
        logger.info("ML model and training data loaded successfully")
    except Exception as e:
        logger.error(f"Could not load ML model: {e}")
        logger.warning("System will use fallback rating system")


# Get current user (from session/cookie)
async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    """Get current user from session"""
    username = request.session.get("username")
    if not username:
        return None
    user = await get_user_by_username(db, username)
    return user


# Home page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, db: AsyncSession = Depends(get_db)):
    """Home page"""
    user = await get_current_user(request, db)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


# Register page
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Register page"""
    user = await get_current_user(request, db)
    return templates.TemplateResponse("register.html", {"request": request, "user": user})


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handle registration"""
    # Check if username already exists
    existing_user = await get_user_by_username(db, username)
    if existing_user:
        user = await get_current_user(request, db)
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "user": user, "error": "Username already exists"}
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Set session
    request.session["username"] = username
    request.session["user_id"] = new_user.id
    
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


# Login page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Login page"""
    user = await get_current_user(request, db)
    return templates.TemplateResponse("login.html", {"request": request, "user": user})


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handle login"""
    user = await authenticate_user(db, username, password)
    if not user:
        current_user = await get_current_user(request, db)
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "user": current_user, "error": "Invalid username or password"}
        )
    
    # Set session
    request.session["username"] = user.username
    request.session["user_id"] = user.id
    
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


# Logout
@app.get("/logout")
async def logout(request: Request):
    """Logout"""
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


# Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """User dashboard"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Get user's application history
    result = await db.execute(
        select(CreditApplication)
        .where(CreditApplication.user_id == current_user.id)
        .order_by(CreditApplication.created_at.desc())
    )
    applications = result.scalars().all()
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "applications": applications
        }
    )


# Application page
@app.get("/apply", response_class=HTMLResponse)
async def apply_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Application page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    return templates.TemplateResponse(
        "apply.html",
        {"request": request, "user": current_user}
    )


@app.post("/apply", response_class=HTMLResponse)
async def submit_application(
    request: Request,
    ssn: str = Form(...),
    full_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    annual_income: float = Form(...),
    monthly_inhand_salary: float = Form(...),
    occupation: str = Form(...),
    month: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit application"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Log application submission
    from app.logger_config import get_logger
    logger = get_logger()
    logger.info(f"New application submitted by user: {current_user.username} (ID: {current_user.id})")
    
    # Calculate credit rating using ML model
    rating_result = calculate_credit_score(
        age=age,
        gender=gender,
        ssn=ssn,
        full_name=full_name,
        annual_income=annual_income,
        monthly_inhand_salary=monthly_inhand_salary,
        occupation=occupation,
        month=month
    )
    
    # Create application record
    application = CreditApplication(
        user_id=current_user.id,
        ssn=ssn,
        full_name=full_name,
        age=age,
        gender=gender,
        annual_income=annual_income,
        monthly_inhand_salary=monthly_inhand_salary,
        occupation=occupation,
        month=month,
        credit_score=rating_result["credit_score"],
        credit_rating=rating_result["credit_rating"],
        interest_rate=rating_result["interest_rate"],
        approved=rating_result["approved"]
    )
    
    db.add(application)
    await db.commit()
    await db.refresh(application)
    
    # Log application result
    logger.info(f"Application #{application.id} processed - Rating: {rating_result['credit_rating']}, "
                f"Score: {rating_result['credit_score']:.2f}, Approved: {rating_result['approved']}")
    
    # Show result page
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "user": current_user,
            "application": application,
            "rating_result": rating_result
        }
    )


# View application details
@app.get("/application/{application_id}", response_class=HTMLResponse)
async def view_application(
    request: Request,
    application_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View application details"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    result = await db.execute(
        select(CreditApplication)
        .where(
            CreditApplication.id == application_id,
            CreditApplication.user_id == current_user.id
        )
    )
    application = result.scalar_one_or_none()
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    rating_result = {
        "credit_score": application.credit_score,
        "credit_rating": application.credit_rating,
        "interest_rate": application.interest_rate,
        "approved": application.approved
    }
    
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "user": current_user,
            "application": application,
            "rating_result": rating_result
        }
    )
