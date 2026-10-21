"""Keep the Streamlit Community Cloud app awake: check if it is up, wake it if asleep, and fail unless it renders."""
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

STREAMLIT_URL = "https://quantamental-screener.streamlit.app"
WAKE_BUTTON = (By.XPATH, "//button[contains(., 'get this app back up')]")
APP_FRAME = (By.CSS_SELECTOR, "iframe[src*='/~/+/']")  # the hosted app, present only once it is running
TIMEOUT = 300


def state(driver):
    """'asleep', 'up' or None (still loading). The app is up only when its own frame has rendered stApp."""
    if driver.find_elements(*WAKE_BUTTON):
        return "asleep"
    frames = driver.find_elements(*APP_FRAME)
    if not frames:
        return None
    driver.switch_to.frame(frames[0])
    try:
        return "up" if driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']") else None
    finally:
        driver.switch_to.default_content()


def wait_for(driver, wanted, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = state(driver)
        if s in wanted:
            return s
        time.sleep(5)
    return None


def main():
    options = Options()
    for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--window-size=1920,1080"):
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(STREAMLIT_URL)
        s = wait_for(driver, ("up", "asleep"), 60)
        print(f"Initial state: {s or 'unknown (page did not load)'}")
        if s == "up":
            return 0
        if s == "asleep":
            driver.find_element(*WAKE_BUTTON).click()
            print("Clicked wake button, waiting for the app to boot...")
        else:
            driver.refresh()
        if wait_for(driver, ("up",), TIMEOUT) == "up":
            print("App is up.")
            return 0
        print(f"App did not come up within {TIMEOUT}s.")
        return 1
    finally:
        driver.quit()


if __name__ == "__main__":
    sys.exit(main())
