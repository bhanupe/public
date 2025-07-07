# --- Prediction for a new image ---
from keras.preprocessing import image
import numpy as np


def predict_image_category(model, image_path, image_height, image_width, class_labels):
    # Load and preprocess the new image
    img = image.load_img(image_path, target_size=(image_height, image_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale

    # Get predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the class label
    predicted_category = class_labels[predicted_class_index]

    return predicted_category, predictions