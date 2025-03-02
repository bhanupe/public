import logging
import random
import string
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Utility Functions
def add(a, b):
    return a + b

def generate_random_number(start=1, end=100):
    """Generates a random number within a given range."""
    return random.randint(start, end)

def generate_random_string(length=8):
    """Generates a random string of given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def validate_numbers(a, b):
    """Validates if inputs are numbers and converts them."""
    try:
        return float(a), float(b)
    except ValueError:
        return None, None

# Routes
@app.route('/')
def home():
    return 'Hello, Sweety!'

@app.route('/add', methods=['POST'])
def add_two_numbers():
    try:
        if request.is_json:
            data = request.get_json()
            a, b = data.get('a'), data.get('b')
        elif request.form:
            a, b = request.form.get('a'), request.form.get('b')
        else:
            return jsonify({"error": "Unsupported media type"}), 400

        if a is None or b is None:
            return jsonify({"error": "Both 'a' and 'b' are required"}), 400

        a, b = validate_numbers(a, b)
        if a is None or b is None:
            return jsonify({"error": "'a' and 'b' must be numbers"}), 400

        result = add(a, b)
        logger.info(f"Addition requested: {a} + {b} = {result}")

        return jsonify({
            "message": f"The sum of {a} and {b} is {result}",
            "result": result
        }), 200
    except Exception as e:
        logger.error(f"Error in /add route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/random-number')
def random_number():
    """Generates a random number and returns it."""
    number = generate_random_number()
    return jsonify({"random_number": number})

@app.route('/random-string')
def random_string():
    """Generates a random string and returns it."""
    rand_str = generate_random_string()
    return jsonify({"random_string": rand_str})

if __name__ == '__main__':
    app.run(debug=True)
