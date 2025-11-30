"""
Logging Configuration
Configure logging to write to file and console
"""
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file path with timestamp
LOG_FILE = os.path.join(LOGS_DIR, f'credit_app_{datetime.now().strftime("%Y%m%d")}.log')


def setup_logging():
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger('credit_app')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - write to log file
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - also print to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger():
    """Get the configured logger"""
    logger = logging.getLogger('credit_app')
    if not logger.handlers:
        setup_logging()
    return logger

