import logging
import sys
import os

def setup_logging(level=logging.INFO):
    """
    Configure logging for the quantamental screener.
    
    Args:
        level: Logging level (default INFO). Set to DEBUG for factor-level troubleshooting.
    """
    # Allow override via environment variable
    env_level = os.getenv('LOG_LEVEL', '').upper()
    if env_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        level = getattr(logging, env_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Optional: file handler for debugging
    if os.getenv('LOG_TO_FILE', 'false').lower() == 'true':
        file_handler = logging.FileHandler('quantamental_screener.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # For debugging factors, set this to DEBUG
    logging.getLogger('factors').setLevel(logging.DEBUG if level <= logging.DEBUG else logging.INFO)
    
    logging.info(f"Logging configured with level {logging.getLevelName(level)}")