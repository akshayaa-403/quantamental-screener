import logging
import sys
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

def setup_logging(level=logging.INFO):
    # (existing code unchanged)
    env_level = os.getenv('LOG_LEVEL', '').upper()
    if env_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        level = getattr(logging, env_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    if os.getenv('LOG_TO_FILE', 'false').lower() == 'true':
        file_handler = logging.FileHandler('quantamental_screener.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('factors').setLevel(logging.DEBUG if level <= logging.DEBUG else logging.INFO)
    logging.info(f"Logging configured with level {logging.getLevelName(level)}")

# Retry decorator for network calls (use with methods that may raise requests.RequestException)
def retry_on_network_error(func):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
    )(func)

logger = logging.getLogger(__name__)

def with_retry(max_attempts=3, min_wait=1, max_wait=10):
    """Decorator for retrying network calls with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} after exception: {retry_state.outcome.exception()}"
        )
    )