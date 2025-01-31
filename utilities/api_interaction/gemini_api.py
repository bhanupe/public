import requests
import time
import os

# Configuration
CONFIG_FILE = "config.txt"  # Configuration file with API details
OUTPUT_FILE = "output.txt"  # File to save API responses
REVIEWS_FILE = "classify_reviews.csv"  # Input file containing reviews
RATE_LIMIT_SLEEP_TIME = 60  # Default sleep time when rate limit is hit
REQUEST_DELAY = 2  # Delay (in seconds) between consecutive API requests
CHUNK_SIZE = 100  # Number of reviews per chunk
MAX_CHUNKS = 3  # Number of chunks to process
MAX_RETRIES = 5  # Maximum retries to avoid infinite loops

def load_api_config(config_file):
    """
    Load API key and URL from a config file instead of .env.
    The config file should have two lines:
    API_KEY=your_api_key
    API_URL=your_api_url
    """
    if not os.path.exists(config_file):
        raise ValueError(f"Configuration file '{config_file}' not found.")

    api_key, api_url = None, None
    with open(config_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("API_KEY="):
                api_key = line.strip().split("=")[1]
            elif line.startswith("API_URL="):
                api_url = line.strip().split("=")[1]

    if not api_key or not api_url:
        raise ValueError("API_KEY or API_URL is missing in the config file.")

    return api_key, api_url

# Load API credentials
API_KEY, API_URL = load_api_config(CONFIG_FILE)

def ask_gemini(api_url, api_key, question, max_retries=MAX_RETRIES):
    """
    Send a question to the Gemini API and handle rate limits.
    Returns only the answer, not unnecessary metadata.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": question}]}]}
    retries = 0

    while retries < max_retries:
        try:
            response = requests.post(f"{api_url}?key={api_key}", headers=headers, json=payload)

            if response.status_code == 200:
                return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            # Handle rate limits and other errors with a retry mechanism
            retry_after = int(response.headers.get("Retry-After", RATE_LIMIT_SLEEP_TIME)) if response.status_code == 429 else RATE_LIMIT_SLEEP_TIME
            print(f"Error {response.status_code}: {response.text}. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            retries += 1

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying in {RATE_LIMIT_SLEEP_TIME} seconds...")
            time.sleep(RATE_LIMIT_SLEEP_TIME)
            retries += 1

    print("Max retries reached. Skipping this request.")
    return None

def read_reviews(file_path, chunk_size, max_chunks):
    """
    Read reviews from a file in chunks and return them as a list.
    """
    print(f"Reading {chunk_size * max_chunks} from {file_path}...")
    reviews = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            next(file)  # Skip header row if present
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
        print(f"Error: Reviews file '{file_path}' not found.")
        raise

def process_reviews(api_url, api_key, reviews, output_file):
    """
    Process each review by sending it to the Gemini API.
    Saves output only once per chunk to reduce file I/O.
    """
    results = []
    for review in reviews:
        question = f"I would like to know if this review is positive or negative. Please respond 'yes' if positive, 'no' if negative for the following review: {review}"
        response = ask_gemini(api_url, api_key, question)
        if response:
            results.append(f"Review: {review} | Response: {response}")

        time.sleep(REQUEST_DELAY)  # Avoid hitting API rate limits

    # Save all responses in one go
    with open(output_file, "a", encoding="utf-8") as file:
        file.write("\n".join(results) + "\n")

# Main Execution
if __name__ == "__main__":
    try:
        print(f"Processing {CHUNK_SIZE * MAX_CHUNKS} reviews in {MAX_CHUNKS} chunks...")

        for chunk_index, review_chunk in enumerate(read_reviews(REVIEWS_FILE, CHUNK_SIZE, MAX_CHUNKS), start=1):
            print(f"Processing chunk {chunk_index} with {len(review_chunk)} reviews...")
            process_reviews(API_URL, API_KEY, review_chunk, OUTPUT_FILE)

        print("Processing completed successfully.")

    except Exception as e:
        print(f"Fatal error occurred: {e}")
