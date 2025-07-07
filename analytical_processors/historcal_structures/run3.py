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
MODEL_SAVE_PATH_FINETUNE = "./saved_model_finetune"
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
# --- Save Model ---
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
import itertools

# Load class labels
with open(CATEGORIES_SAVE_PATH, "r") as f:
    class_labels = json.load(f)

# Define test directory
TEST_DIR = "data/part1/dataset_hist_structures/Dataset_test/Dataset_test_original_1478"

# Get all image paths
all_image_paths = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_image_paths.append(os.path.join(root, file))

if len(all_image_paths) == 0:
    raise ValueError(f"No images found in {TEST_DIR}. Please check the path or image formats.")

# --- Full Test Set Evaluation ---
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    try:
        img = load_img(image_path.numpy().decode('utf-8'), target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = normalization_layer(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path.numpy().decode('utf-8')}: {e}")
        # Return a tensor of zeros with the expected shape and type
        return tf.zeros((*IMAGE_SIZE, 3), dtype=tf.float32)


# Create a TensorFlow dataset from image paths
image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
images_ds = image_paths_ds.map(lambda x: tf.py_function(load_and_preprocess_image, [x], Tout=tf.float32)).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Get true labels
true_labels = [os.path.basename(os.path.dirname(img_path)) for img_path in all_image_paths]

# Predict in batches
predictions = model.predict(images_ds)

# Get predicted labels and confidences
predicted_indices = tf.argmax(predictions, axis=1).numpy()
predicted_labels = [class_labels[i] for i in predicted_indices]
confidences = tf.reduce_max(predictions, axis=1).numpy()

# Create results DataFrame
results = pd.DataFrame({
    "image_path": all_image_paths,
    "predicted_label": predicted_labels,
    "confidence": confidences
})

# Write to CSV
csv_path = "predictions.csv"
results.to_csv(csv_path, index=False)
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

# Calculate skipped images (if any) - This part needs adjustment based on how you handle skipped images in the dataset creation
# For now, we'll assume no images are skipped if the dataset creation with filter works.
# You might need to add a mechanism to track skipped images during dataset creation if necessary.
print(f'number of skipped images: 0')
print(f'skipped images: []')


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# The confusion matrix is already available as `conf_matrix` and class labels as `class_labels`
# Let's analyze the confusion matrix to find the most confused pairs.
# We are looking for high values in the off-diagonal elements.

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
print("Confusion Matrix as DataFrame:")
print(conf_matrix_df)

# Identify top confused pairs (excluding the diagonal)
# Flatten the off-diagonal elements and get their indices
off_diagonal = conf_matrix_df.values.copy()
np.fill_diagonal(off_diagonal, 0)

# Get the indices of the top N largest off-diagonal values
N = 5  # Number of top confused pairs to examine
top_confused_indices = np.argsort(off_diagonal.flatten())[::-1][:N]

# Convert the flat indices back to 2D indices (row, col)
row_indices, col_indices = np.unravel_index(top_confused_indices, off_diagonal.shape)

print(f"\nTop {N} most confused class pairs:")
confused_pairs = []
for r, c in zip(row_indices, col_indices):
    true_class = class_labels[r]
    predicted_class = class_labels[c]
    count = conf_matrix_df.iloc[r, c]
    print(f"True: {true_class}, Predicted: {predicted_class}, Count: {count}")
    confused_pairs.append((true_class, predicted_class, count))

# Filter misclassified images from the results DataFrame
results['true_label'] = results['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
misclassified_df = results[results['predicted_label'] != results['true_label']].copy()

print("\nMisclassified Images DataFrame:")
print(misclassified_df.head())

# Group misclassified images by true and predicted labels
misclassified_grouped = misclassified_df.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
misclassified_grouped = misclassified_grouped.sort_values(by='count', ascending=False)

print("\nMisclassified Image Counts per True/Predicted Pair:")
print(misclassified_grouped.head(10))


# Function to load and display an image
def display_image(image_path, true_label, predicted_label):
    try:
        img = load_img(image_path)
        plt.imshow(img)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image {image_path}: {e}")


# Display a sample of misclassified images for the top confused pairs
sample_size = 3  # Number of images to display for each pair

print(f"\nDisplaying sample misclassified images for the top {N} confused pairs:")
for true_class, predicted_class, count in confused_pairs:
    print(f"\nTrue Class: {true_class}, Predicted Class: {predicted_class} (Count: {count})")
    sample_images = misclassified_df[
        (misclassified_df['true_label'] == true_class) &
        (misclassified_df['predicted_label'] == predicted_class)
        ].sample(min(sample_size, count))  # Sample up to sample_size images

    for index, row in sample_images.iterrows():
        display_image(row['image_path'], row['true_label'], row['predicted_label'])
#
# # ---------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------
# # 1. Define a data augmentation layer
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(height_factor=0.2, width_factor=0.2),
#     layers.RandomContrast(factor=0.2),
#     layers.RandomBrightness(factor=0.2)
# ])
#
#
# # 2. Apply the data augmentation layer to the training dataset
# # Apply augmentation only during training
# def apply_augmentation(image, label):
#     return data_augmentation(image, training=True), label
#
#
# train_ds_augmented = train_ds.map(apply_augmentation)
#
# # 3. Visualize a few examples from the augmented dataset
# print("Visualizing augmented images:")
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds_augmented.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[tf.argmax(labels[i])])
#         plt.axis("off")
# plt.show()
#
# # ---------------------------------------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------
# # --- Unfreeze ViT layers ---
# # Decide how many layers to unfreeze. ViT base has 12 transformer blocks.
# # Let's unfreeze the last 4 transformer blocks.
# num_vit_layers_to_unfreeze = 4
#
# # The ViT model layers are typically structured as:
# # vit.embeddings, vit.encoder.layer[0]...vit.encoder.layer[11], vit.pooler
# # We want to unfreeze the last `num_vit_layers_to_unfreeze` from `vit.encoder.layer`
# vit_encoder_layers = vit_model.vit.encoder.layer
#
# # Freeze all layers first, then unfreeze the last ones
# for layer in vit_encoder_layers:
#     layer.trainable = False
#
# for layer in vit_encoder_layers[-num_vit_layers_to_unfreeze:]:
#     layer.trainable = True
#
# # Also unfreeze the final dense layers in the custom head
# for layer in model.layers[-2:]:  # Unfreeze the Dense layers
#     layer.trainable = True
#
# # Verify trainable layers
# print("Trainable layers in the model:")
# for layer in model.trainable_variables:
#     print(layer.name)
#
# # --- Recompile the model ---
# # Use a lower learning rate for fine-tuning
# fine_tune_learning_rate = 1e-5
#
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
#
# # --- Continue training ---
# fine_tune_epochs = 5  # Train for an additional 5 epochs
#
# print(f"\nContinuing training for {fine_tune_epochs} epochs with fine-tuning...")
#
# history_fine_tune = model.fit(train_ds_augmented,  # Use augmented data for training
#                               epochs=EPOCHS + fine_tune_epochs,  # Total epochs
#                               initial_epoch=EPOCHS,  # Start from where initial training left off
#                               validation_data=val_ds)
#
# # --- Evaluate the fine-tuned model ---
# print("\nEvaluating the fine-tuned model on the validation dataset:")
# loss_fine_tune, accuracy_fine_tune = model.evaluate(val_ds)
# print(f"Validation Loss (Fine-tuned): {loss_fine_tune:.4f}, Accuracy (Fine-tuned): {accuracy_fine_tune:.4f}")
#
# # Compare with previous validation accuracy (if available)
# # Assuming the previous validation accuracy was stored in `accuracy`
# try:
#     print(f"Previous Validation Accuracy: {accuracy:.4f}")
#     print(f"Improvement in Accuracy: {accuracy_fine_tune - accuracy:.4f}")
# except NameError:
#     print("Previous validation accuracy not available for comparison.")
#
# # --- Save Model ---
# model.export(MODEL_SAVE_PATH_FINETUNE)


# --- REPEATING
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

# ... (existing code)

# Load the SavedModel using TFSMLayer
loaded_model = TFSMLayer(MODEL_SAVE_PATH_FINETUNE, call_endpoint='serving_default')

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
import itertools

# Load class labels
with open(CATEGORIES_SAVE_PATH, "r") as f:
    class_labels = json.load(f)

# Define test directory
TEST_DIR = "data/part1/dataset_hist_structures/Dataset_test/Dataset_test_original_1478"

# Get all image paths
all_image_paths = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_image_paths.append(os.path.join(root, file))

if len(all_image_paths) == 0:
    raise ValueError(f"No images found in {TEST_DIR}. Please check the path or image formats.")

# --- Full Test Set Evaluation ---
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd


# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    try:
        img = load_img(image_path.numpy().decode('utf-8'), target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = normalization_layer(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path.numpy().decode('utf-8')}: {e}")
        # Return a tensor of zeros with the expected shape and type
        return tf.zeros((*IMAGE_SIZE, 3), dtype=tf.float32)


# Create a TensorFlow dataset from image paths
image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
images_ds = image_paths_ds.map(lambda x: tf.py_function(load_and_preprocess_image, [x], Tout=tf.float32)).batch(
    BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Get true labels
true_labels = [os.path.basename(os.path.dirname(img_path)) for img_path in all_image_paths]

# Predict in batches
predictions = model.predict(images_ds)

# Get predicted labels and confidences
predicted_indices = tf.argmax(predictions, axis=1).numpy()
predicted_labels = [class_labels[i] for i in predicted_indices]
confidences = tf.reduce_max(predictions, axis=1).numpy()

# Create results DataFrame
results = pd.DataFrame({
    "image_path": all_image_paths,
    "predicted_label": predicted_labels,
    "confidence": confidences
})

# Write to CSV
csv_path = "predictions_before.csv"
results.to_csv(csv_path, index=False)
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

# Calculate skipped images (if any) - This part needs adjustment based on how you handle skipped images in the dataset creation
# For now, we'll assume no images are skipped if the dataset creation with filter works.
# You might need to add a mechanism to track skipped images during dataset creation if necessary.
print(f'number of skipped images: 0')
print(f'skipped images: []')

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# The confusion matrix is already available as `conf_matrix` and class labels as `class_labels`
# Let's analyze the confusion matrix to find the most confused pairs.
# We are looking for high values in the off-diagonal elements.

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
print("Confusion Matrix as DataFrame:")
print(conf_matrix_df)

# Identify top confused pairs (excluding the diagonal)
# Flatten the off-diagonal elements and get their indices
off_diagonal = conf_matrix_df.values.copy()
np.fill_diagonal(off_diagonal, 0)

# Get the indices of the top N largest off-diagonal values
N = 5  # Number of top confused pairs to examine
top_confused_indices = np.argsort(off_diagonal.flatten())[::-1][:N]

# Convert the flat indices back to 2D indices (row, col)
row_indices, col_indices = np.unravel_index(top_confused_indices, off_diagonal.shape)

print(f"\nTop {N} most confused class pairs:")
confused_pairs = []
for r, c in zip(row_indices, col_indices):
    true_class = class_labels[r]
    predicted_class = class_labels[c]
    count = conf_matrix_df.iloc[r, c]
    print(f"True: {true_class}, Predicted: {predicted_class}, Count: {count}")
    confused_pairs.append((true_class, predicted_class, count))

# Filter misclassified images from the results DataFrame
results['true_label'] = results['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
misclassified_df = results[results['predicted_label'] != results['true_label']].copy()

print("\nMisclassified Images DataFrame:")
print(misclassified_df.head())

# Group misclassified images by true and predicted labels
misclassified_grouped = misclassified_df.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
misclassified_grouped = misclassified_grouped.sort_values(by='count', ascending=False)

print("\nMisclassified Image Counts per True/Predicted Pair:")
print(misclassified_grouped.head(10))


# Function to load and display an image
def display_image(image_path, true_label, predicted_label):
    try:
        img = load_img(image_path)
        plt.imshow(img)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image {image_path}: {e}")


# Display a sample of misclassified images for the top confused pairs
sample_size = 3  # Number of images to display for each pair

print(f"\nDisplaying sample misclassified images for the top {N} confused pairs:")
for true_class, predicted_class, count in confused_pairs:
    print(f"\nTrue Class: {true_class}, Predicted Class: {predicted_class} (Count: {count})")
    sample_images = misclassified_df[
        (misclassified_df['true_label'] == true_class) &
        (misclassified_df['predicted_label'] == predicted_class)
        ].sample(min(sample_size, count))  # Sample up to sample_size images

    for index, row in sample_images.iterrows():
        display_image(row['image_path'], row['true_label'], row['predicted_label'])
