import itertools
import os
import tensorflow as tf
from transformers import TFViTModel
from tensorflow.keras import layers, models
import json

# --- Configuration ---
DATASET_DIR = "data/part1/dataset_hist_structures/Stuctures_Dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123
MODEL_SAVE_PATH = "./saved_model"
CATEGORIES_SAVE_PATH = "./categories.json"

# --- Load Dataset ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    label_mode="categorical",
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    label_mode="categorical",
    batch_size=BATCH_SIZE
)

# --- Normalization ---
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# --- Prefetch ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Extract number of classes ---
num_classes = len(class_names)

# --- Save class labels ---
with open(CATEGORIES_SAVE_PATH, "w") as f:
    json.dump(class_names, f)

# --- Load ViT base model ---
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224", from_pt=True)
vit_model.trainable = False  # Freeze the model


# --- Custom wrapper for ViT ---
class ViTWrapper(tf.keras.layers.Layer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.permute = layers.Permute((3, 1, 2))

    def call(self, inputs):
        x = self.permute(inputs)
        outputs = self.base_model(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

    def get_config(self):
        config = super().get_config()
        return config  # base_model excluded


# --- Model definition ---
inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = ViTWrapper(vit_model)(inputs)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
# model = models.Model(inputs, outputs)
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # --- Train ---
# model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
#
# # --- Save Model ---
# model.export(MODEL_SAVE_PATH)


import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

# ... (existing code)

# Load the SavedModel using TFSMLayer
loaded_model = TFSMLayer(MODEL_SAVE_PATH, call_endpoint='serving_default')

# Create a new model with the loaded SavedModel layer
model = models.Model(inputs, outputs)

# Compile the model (optional if you already compiled it when saving)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Evaluation on validation dataset ---
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# --- Prediction on 10 random images from test set ---
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# Load class labels
with open(CATEGORIES_SAVE_PATH, "r") as f:
    class_labels = json.load(f)

# Define test directory
TEST_DIR = "data/part1/dataset_hist_structures/Dataset_test/Dataset_test_original_1478"

# Get 10 random image paths
all_images = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, file))

if len(all_images) == 0:
    raise ValueError(f"No images found in {TEST_DIR}. Please check the path or image formats.")

# --- Full Test Set Evaluation ---
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

true_labels = []
predicted_labels = []
skipped_images = []
results = []
for img_path in all_images:
    # Derive true label from folder name
    true_label = os.path.basename(os.path.dirname(img_path))
    try:
        img = load_img(img_path, target_size=IMAGE_SIZE)
    except:
        skipped_images.append(img_path)
        continue
    img_array = img_to_array(img)
    img_array = normalization_layer(tf.expand_dims(img_array, axis=0))

    predictions = model.predict(img_array)
    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_label = class_labels[predicted_index]
    # Display the image and prediction
    # plt.imshow(img)
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis("off")
    # plt.show()
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

    results.append({
        "image_path": img_path,
        "predicted_label": predicted_label,
        "confidence": float(tf.reduce_max(predictions[0]).numpy())
    })

# Write to CSV
csv_path = "predictions.csv"
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"Predictions saved to {csv_path}")

# --- Confusion Matrix and Report ---
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

fmt = 'd'
thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(f'number of skipped images: {len(skipped_images)}')
print(f'skipped images: {skipped_images}')
