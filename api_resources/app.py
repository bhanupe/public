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

@app.route('/generate-number', methods=['GET'])
def generate_random_number():
    start = int(request.args.get('start', 1))
    end = int(request.args.get('end', 100))
    random_number = random.randint(start, end)
    logger.info(f'Generated random number: {random_number}')
    return jsonify({"random_number": random_number})

@app.route('/generate-string', methods=['GET'])
def generate_random_string():
    length = int(request.args.get('length', 10))
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    logger.info(f'Generated random string: {random_string}')
    return jsonify({"random_string": random_string})

@app.route('/format-string', methods=['GET'])
def format_string():
    text = request.args.get('text', '')
    format_type = request.args.get('format', 'snake_case')
    formatted_text = ''

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

@app.route('/generate-strings', methods=['PUT'])
def generate_strings():
    data = request.json
    num_strings = data.get('num_strings', 5)
    length = data.get('length', 10)
    strings = []
    total_length = 0

    for _ in range(num_strings):
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        strings.append(random_string)
        total_length += len(random_string)
        logger.info(f'Generated random string: {random_string}')

    return jsonify({"strings": strings, "total_length": total_length})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
