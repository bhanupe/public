import requests
import time
import os
from pprint import pprint

# Configuration
CONFIG_FILE = "config.txt"  # File containing API credentials
LOG_FILE = "interaction_log.txt"  # File to log interactions
OUTPUT_FILE = "output.txt"  # File to save API responses
REVIEWS_FILE = "classify_reviews.csv"  # Input file containing reviews
RATE_LIMIT_SLEEP_TIME = 60  # Default sleep time when rate limit is hit
REQUEST_DELAY = 2  # Delay (in seconds) between consecutive API requests
CHUNK_SIZE = 100  # Number of reviews per chunk
MAX_CHUNKS = 3  # Number of chunks to process (set to 1 for testing, 3 for 300 reviews)
MAX_RETRIES = 5  # Maximum retries to avoid infinite loops

# Load configuration from text file
def load_config(file_path):
    """
    Load API configuration from a text file.
    """
    config = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split("=", 1)
                config[key.strip()] = value.strip()
    except FileNotFoundError:
        raise ValueError(f"Configuration file '{file_path}' not found.")
    return config

config = load_config(CONFIG_FILE)
API_URL = config.get("API_URL")
API_KEY = config.get("API_KEY")

# Validate configuration
if not API_KEY or not API_URL:
    raise ValueError("API_KEY or API_URL is missing in the config.txt file.")

def ask_gemini(api_url, api_key, question, max_retries=MAX_RETRIES):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": question}]}]}
    retries = 0
    while retries < max_retries:
        try:
            print(f"\nSending question to Gemini API (Attempt {retries + 1}):")
            pprint(payload)
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                print("\nResponse received from Gemini API:")
                pprint(response_data)
                return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RATE_LIMIT_SLEEP_TIME))
                print(f"\nRate limit hit. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                retries += 1
            else:
                print(f"\nError from Gemini API: {response.status_code}, {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed due to an exception: {e}")
            retries += 1
            time.sleep(RATE_LIMIT_SLEEP_TIME)
    print("\nMax retries reached. Skipping this request.")
    return None

def log_interaction(question, response):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n\n")

def save_output(review, response):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(f"Review: {review} | Response: {response}\n")

def read_reviews(file_path, chunk_size, max_chunks):
    reviews = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i >= chunk_size * max_chunks:
                    break
                if i % chunk_size == 0 and i > 0:
                    yield reviews
                    reviews = []
                reviews.append(line.strip())
            if reviews:
                yield reviews
    except FileNotFoundError:
        print(f"\nError: Reviews file '{file_path}' not found.")
        raise

def process_reviews(api_url, api_key, reviews):
    for i, review in enumerate(reviews, start=1):
        question = f"I would like to know if this review is positive or negative. Please respond yes if positive, no if negative for the following review: {review}"
        print(f"\nProcessing review {i}: {review}")
        response = ask_gemini(api_url, api_key, question)
        if response:
            save_output(review, response)
            log_interaction(question, response)
        else:
            print(f"\nFailed to process review {i}. No response from Gemini API.")
        time.sleep(REQUEST_DELAY)

if __name__ == "__main__":
    try:
        print(f"\nStarting review processing with CHUNK_SIZE={CHUNK_SIZE} and MAX_CHUNKS={MAX_CHUNKS}...")
        for chunk_index, review_chunk in enumerate(read_reviews(REVIEWS_FILE, CHUNK_SIZE, MAX_CHUNKS), start=1):
            print(f"\nProcessing chunk {chunk_index} with {len(review_chunk)} reviews...")
            process_reviews(API_URL, API_KEY, review_chunk)
        print("\nProcessing completed successfully.")
    except Exception as e:
        print(f"\nError occurred: {e}")
