from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello, Sweety!'


def add(a, b):
    return a + b


@app.route('/add', methods=['POST'])
def add_two_numbers():
    if request.is_json:
        data = request.get_json()
        return {"message": "The sum of 2 values a and b is as followed", "result": add(data['a'], data['b'])}, 200
    elif request.form:
        data = request.form
        # Process form data
        return {"message": "Form data received", "data": data}, 200
    else:
        return {"error": "Unsupported media type"}, 400


if __name__ == '__main__':
    app.run(debug=True)
