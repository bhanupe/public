import os
import time

import requests

# Configuration
CONFIG_FILE = "config.txt"  # Configuration file with API details
OUTPUT_FILE = "output.txt"  # File to save API responses
REVIEWS_FILE = "classify_reviews.csv"  # Input file containing reviews
RATE_LIMIT_SLEEP_TIME = 60  # Default sleep time when rate limit is hit
REQUEST_DELAY = 4  # Delay (in seconds) between consecutive API requests
CHUNK_SIZE = 100  # Number of reviews per chunk
MAX_CHUNKS = 250  # Number of chunks to process
MAX_REVIEWS = 5000  # Maximum reviews to process
MAX_RETRIES = 3  # Maximum retries to avoid infinite loops


def load_api_config(config_file):
    """ Load API key and URL from config file instead of .env
        The config file should have two lines:
        API_KEY=YOUR_API_KEY
        API_URL=YOUR_API_URL
    """
    if not os.path.exists(config_file):
        raise ValueError(f"Config file '{config_file}' not found.")

    api_key, api_url = None, None
    with open(config_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("API_KEY="):
                api_key = line.strip().split("=", 1)[1]
            elif line.startswith("API_URL="):
                api_url = line.strip().split("=", 1)[1]

    if not api_key or not api_url:
        raise ValueError("Missing API_KEY or API_URL in config file.")
    if not api_url.startswith("http"):
        raise ValueError(f"Invalid API URL: {api_url}")

    return api_key, api_url


def process_review_text(review):
    review = review.split('\t')[0]
    return review


def ask_gemini(api_url, api_key, question, max_retries=MAX_RETRIES):
    """ Sends a request to Gemini API and handles rate limits.
        Returns only the answer, not unnecessary metadata.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": question}]}]}
    full_url = f"{api_url}?key={api_key}"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(full_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            elif response.status_code == 429:  # Handle Rates limit hit
                print(f"⚠️ Rate limit exceeded. Waiting {RATE_LIMIT_SLEEP_TIME}s... ({attempt}/{max_retries})")
                time.sleep(RATE_LIMIT_SLEEP_TIME)
            else:
                print(
                    f"❌ API Error {response.status_code}: {response.json().get('error', {}).get('message', 'Unknown error')}")
                break  # Exit on non-retryable errors

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network Error: {e}. Retrying in {RATE_LIMIT_SLEEP_TIME}s...")
            time.sleep(RATE_LIMIT_SLEEP_TIME)

    print("❌ Max retries reached. Skipping request.")
    return None


def read_reviews(file_path, chunk_size, max_reviews):
    """ Reads reviews from file in chunks and return them as a list. """
    print(f"📖 Reading up to {max_reviews} reviews from {file_path}...")
    lines_read = 0
    chunk = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            next(file)  # Skip header row
            for line in file:
                if lines_read >= max_reviews:
                    break

                cleaned_review = process_review_text(line)
                if not cleaned_review:
                    continue

                chunk.append(cleaned_review)
                lines_read += 1

                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []

        if chunk:
            yield chunk

    except FileNotFoundError:
        print(f"❌ Error: Reviews file '{file_path}' not found.")
        raise


def process_reviews(api_url, api_key, reviews, output_file):
    """ Process each review by sending it to the Gemini API.
        Saves output only once per chunk to reduce I/O.
    """
    results = []
    for review in reviews:
        question = f"Is this review positive or negative? Reply 'yes' for positive, 'no' for negative: {review}"
        response = ask_gemini(api_url, api_key, question)

        if response:
            results.append(f"Review: {review} | Response: {response}")
        else:
            print("⚠️ Skipped a review due to API error.")

        # Adding a request delay to prevent overwhelming the API and avoid hitting rate limits
        time.sleep(REQUEST_DELAY)

    # Save all responses in one go
    with open(output_file, "a", encoding="utf-8") as file:
        file.write("\n".join(results) + "\n")


# Main Execution
if __name__ == "__main__":
    try:
        # Load API credentials
        API_KEY, API_URL = load_api_config(CONFIG_FILE)
        print(f"🚀 Processing up to {MAX_REVIEWS} reviews in {MAX_CHUNKS} chunks...")

        for chunk_index, review_chunk in enumerate(read_reviews(REVIEWS_FILE, CHUNK_SIZE, MAX_REVIEWS), start=1):
            print(f"📦 Processing chunk {chunk_index} with {len(review_chunk)} reviews...")
            process_reviews(API_URL, API_KEY, review_chunk, OUTPUT_FILE)

        print("✅ Processing completed.")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
