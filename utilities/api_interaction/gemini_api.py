import requests
import time
import os

# Configuration
LOG_FILE = "interaction_log.txt"  # File to log interactions (questions and responses).
OUTPUT_FILE = "output.txt"  # File to save API responses.
REVIEWS_FILE = "classify_reviews.csv"  # Input file containing reviews.
CONFIG_FILE = "config.txt"  # Configuration file containing API_KEY and API_URL.
RATE_LIMIT_SLEEP_TIME = 60  # Default sleep time (in seconds) for rate limit handling.
REQUEST_DELAY = 1  # Delay (in seconds) between consecutive API requests.
NUM_REVIEWS_TO_PROCESS = 100  # Number of reviews to process (adjustable).

# Load configuration from a text file
def load_config(file_path):
    """
    Load API_KEY and API_URL from a configuration text file.
    Parameters:
        file_path (str): Path to the configuration file.
    Returns:
        dict: A dictionary containing API_KEY and API_URL.
    """
    config = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split("=")
                config[key.strip()] = value.strip()
    except FileNotFoundError:
        raise Exception(f"Configuration file '{file_path}' not found.")
    except ValueError:
        raise Exception(f"Invalid format in configuration file '{file_path}'. Each line must be in 'KEY=VALUE' format.")
    return config

# Load API configuration
config = load_config(CONFIG_FILE)
API_KEY = config.get("API_KEY")
API_URL = config.get("API_URL")

# Validate configuration
if not API_KEY or not API_URL:
    raise ValueError("API_KEY or API_URL is missing in the configuration file.")

# Functions
def ask_gemini(api_url, api_key, question):
    """
    Send a question to the Gemini API and handle rate limits.
    Parameters:
        api_url (str): API endpoint URL.
        api_key (str): API key for authentication.
        question (str): The question to send to the API.
    Returns:
        str: The response text from the API or None if an error occurs.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": question}]
            }
        ]
    }
    
    while True:
        try:
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RATE_LIMIT_SLEEP_TIME))
                print(f"Rate limit reached. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

def log_interaction(question, response):
    """
    Log the interaction (question and response) to a file.
    Parameters:
        question (str): The question sent to the API.
        response (str): The API response.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n\n")

def save_output(response):
    """
    Save the API response to a text file.
    Parameters:
        response (str): The response to save.
    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(f"Gemini Response:\n{response}\n\n")

def read_reviews(file_path, num_reviews):
    """
    Read reviews from a file and return a limited number of them as a list.
    Parameters:
        file_path (str): Path to the reviews file.
        num_reviews (int): Number of reviews to read.
    Returns:
        list: A list of reviews (strings).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()][:num_reviews]
    except FileNotFoundError:
        raise Exception(f"Reviews file '{file_path}' not found.")

def process_reviews(api_url, api_key, reviews):
    """
    Process each review by sending it to the Gemini API.
    Parameters:
        api_url (str): API endpoint URL.
        api_key (str): API key for authentication.
        reviews (list): List of reviews to process.
    """
    for i, review in enumerate(reviews, start=1):
        question = f"I would like to know if this review is positive or negative. Please respond yes if positive, no if negative for the following review: {review}"
        print(f"Processing review {i}: {review}")

        response = ask_gemini(api_url, api_key, question)
        if response:
            print(f"Response: {response}")
            log_interaction(question, response)
            save_output(response)
        
        time.sleep(REQUEST_DELAY)

# Test cases
def run_tests():
    """
    Run test cases to validate script functionality.
    """
    try:
        print("Testing API connection...")
        test_question = "What is artificial intelligence?"
        response = ask_gemini(API_URL, API_KEY, test_question)
        assert response is not None, "API response is None."
        print("API Test Passed.")
    except Exception as e:
        print(f"API Test Failed: {e}")

    try:
        print("Testing review reading...")
        reviews = read_reviews(REVIEWS_FILE, NUM_REVIEWS_TO_PROCESS)
        assert len(reviews) > 0, "No reviews were read."
        print("Review Reading Test Passed.")
    except Exception as e:
        print(f"Review Reading Test Failed: {e}")

# Main execution
if __name__ == "__main__":
    try:
        # Read reviews from the input file
        reviews = read_reviews(REVIEWS_FILE, NUM_REVIEWS_TO_PROCESS)

        # Process reviews using the Gemini API
        process_reviews(API_URL, API_KEY, reviews)

        # Run tests
        run_tests()
    except Exception as e:
        print(f"Error: {e}")
