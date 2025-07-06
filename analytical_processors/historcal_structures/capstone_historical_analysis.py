# Install if not already present
# pip install opencv-python tensorflow matplotlib seaborn scikit-surprise "numpy<2"

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import ssl
from datetime import datetime


def get_datetime():
    now = datetime.now()

    # It is also possible to format the output:
    return now.strftime("%Y%m%d%H%M%S")


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

show=True
folder = get_datetime()
create_folder(folder)

def save_show(plot, name):
    plot.savefig(f'{folder}/{name}.png')
    if show:
        plot.show()


# Fix for SSL: CERTIFICATE_VERIFY_FAILED
ssl._create_default_https_context = ssl._create_unverified_context


# Paths to image dataset
data_dir = 'data/part1/dataset_hist_structures/Stuctures_Dataset'  # Replace with actual path after unzip
categories = os.listdir(data_dir)
if '.DS_Store' in categories:
    categories.remove('.DS_Store')
# Sample image plots
for category in categories:
    path = os.path.join(data_dir, category)
    images = random.sample(os.listdir(path), 8)
    plt.figure(figsize=(12, 5))
    for i, img_file in enumerate(images):
        img = cv2.imread(os.path.join(path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
    save_show(plt, f'{category}_SampleImagePlots')


# Exploratory Data Analysis (EDA)
# ---
# Analyze the distribution of images per category to understand the dataset balance.

# Create a dataframe with image paths and labels
image_paths = []
labels = []
for category in categories:
    cat_path = os.path.join(data_dir, category)
    for image_file in os.listdir(cat_path):
        image_paths.append(os.path.join(cat_path, image_file))
        labels.append(category)

df = pd.DataFrame({'path': image_paths, 'label': labels})

# Plot the distribution of images per category
plt.figure(figsize=(12, 6))
sns.countplot(y=df['label'], order=df['label'].value_counts().index)
plt.title('Image Distribution per Category')
plt.xlabel('Number of Images')
plt.ylabel('Category')
plt.tight_layout()
save_show(plt, f'ImageDistributionPerCategory')


# 2. Transfer Learning
# Instead of training a model from scratch, you reuse a pre-trained model (already trained on a large dataset like ImageNet).
#
# You then fine-tune it or add new layers for your specific task.
#
# Why use it?
#
# Saves time, computational resources, and data.
#
# Helps get good accuracy even with smaller datasets.
#
# 3. MobileNetV2 Architecture
# A lightweight, efficient CNN designed by Google for mobile and edge devices.
#
# Pretrained on ImageNet (millions of general images).
#
# Known for fast inference and small model size, with good accuracy.
# So, putting it all together:
# A deep learning model using transfer learning based on the MobileNetV2 architecture means:

# You start with a pre-trained MobileNetV2 model that already knows general image features.
#
# You freeze its early layers (so it keeps what it already learned).
#
# You add custom layers on top (for your task, like classifying historical structures).
#
# You train only the new layers, or optionally fine-tune some layers of MobileNetV2 as well.

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# By examining this plot, you can see how the training and validation accuracies change over time.
#
# If the training accuracy is much higher than the validation accuracy, it indicates overfitting (the model is memorizing the training data but not generalizing well to unseen data).
# If both accuracies are increasing and close to each other, the model is learning well.
# If both accuracies plateau or decrease, it might indicate that the model has stopped learning or that the learning rate is too high.

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
save_show(plt, f'TrainingVsValidationAccuracy')

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
save_show(plt, 'TrainingVsValidationLoss')

# --- Save Model ---
# This creates a file named structure_classifier_model.h5 containing:
#
# Model architecture
#
# Weights
#
# Optimizer state
#
# Training config
model.save("structure_classifier_model.h5")


# Load the saved model
model = load_model("structure_classifier_model.h5")

TEST_DIR = 'data/part1/dataset_hist_structures/Dataset_test'

# Get 10 random image paths
all_images = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, file))

y_true = []
y_pred = []

for img_path in all_images:
    try:
        # Derive true label from folder name
        true_label = os.path.basename(os.path.dirname(img_path))
        # This line loads an image from the path specified by img_path and resizes it to 224x224 pixels. This is the input size required by the MobileNetV2 model used earlier.
        img = image.load_img(img_path, target_size=(224, 224))
        # This converts the loaded image into a NumPy array. This is necessary because neural networks process data in array format.
        img_array = image.img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0) / 255.0: This line does two things:
        # np.expand_dims(img_array, axis=0): Adds an extra dimension to the array. This is because the model expects a batch of images, even if we are only providing one. The added dimension represents the batch size (in this case, 1).
        # / 255.0: This normalizes the pixel values by dividing by 255.0. Pixel values are typically in the range of 0 to 255, and normalizing them to a range of 0 to 1 helps the model train more effectively.
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        # The ImageDataGenerator used for training (train_generator) automatically assigns an integer index to each class (category) it finds in the dataset. This line retrieves that mapping, where the keys are the class names and the values are the integer indices.
        # labels = dict((v, k) for k, v in class_indices.items()): This line creates a new dictionary called labels. It reverses the mapping from class_indices, so that the keys are now the integer indices and the values are the class names. This makes it easy to look up the class name using the predicted index from the model.
        class_indices = train_generator.class_indices
        labels = dict((v, k) for k, v in class_indices.items())
        pred_class = labels[np.argmax(prediction)]

        print(f"{img_path} ➜ Predicted Class: {pred_class} ➜ True label: {true_label}")
        y_true.append(true_label)
        y_pred.append(pred_class)
    except:
        print(f"skipping {img_path}")
        continue

# --- Confusion Matrix ---
class_names = list(train_generator.class_indices.keys())
cm = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
save_show(plt, f'ConfusionMatrix')

