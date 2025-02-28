from flask import Flask, request, jsonify
import random
import string
import logging
import re

app = Flask(__name__)

# Logger setup
def setup_logger(log_file="api_logs.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

logger = setup_logger()

# Reusable function: Generate a random string
def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# Reusable function: Sum of digits in a string
def sum_of_digits_in_string(text):
    return sum(int(char) for char in text if char.isdigit())

# Reusable function: Calculate string length
def calculate_string_length(text):
    return len(text)

@app.route('/generate-number', methods=['GET'])
def generate_random_number():
    start = int(request.args.get('start', 1))
    end = int(request.args.get('end', 100))
    return jsonify({"random_number": random.randint(start, end)})

@app.route('/generate-string', methods=['GET'])
def get_random_string():
    length = int(request.args.get('length', 10))
    return jsonify({"random_string": generate_random_string(length)})

@app.route('/format-string', methods=['GET'])
def format_string():
    text = request.args.get('text', '')
    format_type = request.args.get('format', 'snake_case')

    if format_type == "snake_case":
        formatted_text = text.lower().replace(" ", "_")
    elif format_type == "camelCase":
        words = re.split(r'\s+', text.strip())
        formatted_text = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    else:
        return jsonify({"error": "Invalid format type"}), 400

    return jsonify({"formatted_text": formatted_text})

@app.route('/log-message', methods=['POST'])
def log_message():
    log_level = request.json.get('level', 'info').lower()
    message = request.json.get('message', 'No message provided')

    if log_level == "info":
        logger.info(message)
    elif log_level == "error":
        logger.error(message)
    elif log_level == "debug":
        logger.debug(message)
    else:
        return jsonify({"error": "Invalid log level"}), 400

    return jsonify({"status": "Logged", "log_level": log_level, "message": message})

@app.route('/process-string', methods=['PUT'])
def process_string():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    return jsonify({
        "original_text": text,
        "sum_of_digits": sum_of_digits_in_string(text),
        "string_length": calculate_string_length(text)
    })

if __name__ == '__main__':
    app.run(debug=True)
