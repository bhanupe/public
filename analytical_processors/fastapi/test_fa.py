from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import pandas as pd

from analytical_processors.wrangling.prepare import encode

app = Flask(__name__)

# # Load the model and scaler
model = tf.keras.models.load_model('../loan_default_keras_model.h5')
scaler = joblib.load('../scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json(force=True)
        data_new = pd.DataFrame.from_dict(data)
        print(data_new)
        data_new_encoded = encode(data_new, int, 'data_n_encoded.csv')
        # Step 1: Prepare Features (X) and Target (y)
        X_new = data_new_encoded.drop('not.fully.paid', axis=1)  # Drop target column from features # Input features
        y_new = data_new_encoded['not.fully.paid']

        # Apply the scalar transformation (if needed)
        new_data_scaled = scaler.transform(X_new)

        # Now you can use it to predict
        predictions = model.predict(new_data_scaled)  # (X_new_scaled = new customer data)

        # Convert probabilities to 0 or 1
        predicted_classes = (predictions > 0.5).astype(int)

        # Create a DataFrame for easy visualization
        result_df = pd.DataFrame({
            'default_probability_percent': predictions.flatten() * 100,
            'predicted_class': predicted_classes.flatten()
        })
        print(result_df)  # See first 10 predictions

        return result_df.to_json(orient='records'), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
