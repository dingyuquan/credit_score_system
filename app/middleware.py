from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Session middleware configuration
SECRET_KEY = "your-session-secret-key-change-in-production"
