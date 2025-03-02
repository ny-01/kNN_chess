import json
import time
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def fen_to_url(fen):
    """
    Convert a FEN string into the URL format used by Lichess analysis board.
    This replaces spaces with underscores.
    """
    return fen.replace(" ", "_")

# Set up Firefox (visible browser so you can see the action).
options = Options()
# Comment out headless so you see the browser.
# options.add_argument("--headless")
driver = webdriver.Firefox(options=options)
wait = WebDriverWait(driver, 15)

results = []

# Process the simplified database (only first 10 positions for testing).
with open("simplified_db.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        fen = data["fen"]
        expected_eval = data["eval"]
        print(f"Processing position {i+1} FEN: {fen}")

        # Construct the URL for Lichess's analysis board.
        url = "https://lichess.org/analysis/fromPosition/" + fen_to_url(fen)
        print("Navigating to:", url)
        driver.get(url)
        
        # Wait a few seconds for the board to load.
        time.sleep(3)
        
        # Press the "L" key to toggle local evaluation.
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys("l")
        
        # Wait a few seconds for the evaluation to update.
        time.sleep(3)
        
        # Wait for the evaluation element to appear.
        try:
            # According to the HTML you provided, the numerical evaluation is inside a <pearl> element within .ceval.
            eval_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ceval pearl")))
            displayed_text = eval_elem.text.strip()
            try:
                # Lichess typically displays evaluations like "+16.6" or "-0.56"
                displayed_eval = float(displayed_text.replace("+", ""))
            except ValueError:
                displayed_eval = None
        except Exception as e:
            print(f"Could not get eval for position {i+1}: {e}")
            displayed_eval = None

        results.append({
            "fen": fen,
            "expected_eval": expected_eval,
            "displayed_eval": displayed_eval
        })

        # For a sanity check, process only the first 10 positions.
        if i >= 9:
            break

driver.quit()

# Print the results.
for r in results:
    print(r)