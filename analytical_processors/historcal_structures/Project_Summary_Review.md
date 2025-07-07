# Project Summary: Historical Structures Image Classification

This document provides a step-by-step summary of the `capstone_historical_analysis.py` script and offers suggestions for improvement.

---

## I. Project Operations: Step-by-Step

The project is designed to classify images of historical architectural features using a deep learning model. The process is executed as follows:

### Step 1: Environment Setup and Initialization
- **Dependencies**: The script requires libraries such as TensorFlow, OpenCV, Matplotlib, Seaborn, and Scikit-learn.
- **Folder Creation**: A timestamped folder is created to store all output plots and artifacts, ensuring that results from each run are saved separately.
- **SSL Configuration**: A workaround is applied to handle SSL certificate verification issues, which can arise when downloading pre-trained model weights.

### Step 2: Data Loading and Exploratory Data Analysis (EDA)
- **Data Loading**: The script identifies all categories of historical structures by listing the subdirectories in the dataset folder.
- **Sample Visualization**: For each category, a random sample of 8 images is displayed and saved as a plot. This provides a quick visual check of the dataset's contents.
- **Data Distribution Analysis**: The script counts the number of images in each category and visualizes this distribution using a bar plot. This step is crucial for identifying any class imbalance, which could affect model performance.

### Step 3: Model Architecture and Definition (Transfer Learning)
- **Base Model**: The project leverages **Transfer Learning** by using the **MobileNetV2** architecture, pre-trained on the ImageNet dataset. This allows the model to use pre-existing knowledge of general image features.
- **Freezing Layers**: The layers of the base MobileNetV2 model are frozen (`trainable = False`) to prevent their weights from being updated during the initial training phase.
- **Custom Head**: A custom classification head is added on top of the base model. This consists of:
    - A `GlobalAveragePooling2D` layer to reduce the feature map dimensions.
    - A `Dropout` layer (with a 50% rate) to prevent overfitting.
    - A `Dense` layer with 128 neurons and a 'relu' activation function.
    - A final `Dense` output layer with a 'softmax' activation function to produce a probability distribution across the categories.

### Step 4: Model Compilation and Training
- **Compilation**: The model is compiled with the 'adam' optimizer, 'categorical_crossentropy' loss function (standard for multi-class classification), and 'accuracy' as the performance metric.
- **Data Augmentation and Generators**: An `ImageDataGenerator` is used to rescale pixel values to a [0, 1] range and split the data into training (80%) and validation (20%) sets.
- **Training Process**: The model is trained for a maximum of 20 epochs. An `EarlyStopping` callback is used to monitor the validation accuracy and stop the training process if it doesn't improve for 5 consecutive epochs, which saves time and prevents overfitting.

### Step 5: Performance Visualization and Model Saving
- **Metrics Plotting**: After training, the script plots the training and validation accuracy and loss over epochs. These plots are essential for diagnosing issues like overfitting or underfitting.
- **Model Saving**: The trained model, including its architecture and learned weights, is saved to a single file (`structure_classifier_model.h5`).

### Step 6: Model Evaluation on Test Data
- **Model Loading**: The saved model is loaded back into memory for evaluation.
- **Prediction on Test Set**: The model predicts the class for each image in a separate test directory. For each prediction, it prints the predicted class alongside the true label.
- **Confusion Matrix**: A confusion matrix is generated to provide a detailed view of the model's performance, showing which classes are often confused with others. This matrix is visualized as a heatmap and saved.

---

## II. Suggestions for Improvement

The project provides a solid foundation for image classification. The following suggestions could further enhance its robustness, performance, and maintainability.

### 1. Enhance Data Augmentation
The current `ImageDataGenerator` only rescales images. To improve model generalization and make it more robust to variations in real-world images, consider adding more augmentation techniques to the **training generator**:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,      # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift width
    height_shift_range=0.2, # Randomly shift height
    shear_range=0.2,        # Apply shear transformations
    zoom_range=0.2,         # Randomly zoom in
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'
)
```

### 2. Implement Hyperparameter Tuning
The script uses fixed values for hyperparameters like learning rate, dropout rate, and the number of neurons in the dense layer. These could be optimized using a systematic approach like **KerasTuner** or **Grid Search** to potentially find a better-performing model configuration.

### 3. Refine Code Structure and Modularity
The script is written in a linear fashion. Refactoring the code into functions would improve readability and reusability. For example:
- `build_model(num_classes)`: A function to construct and compile the model.
- `train_model(model, train_gen, val_gen)`: A function to handle the training loop.
- `evaluate_model(model, test_dir, class_indices)`: A function for evaluation and generating the confusion matrix.

### 4. Improve Error Handling
The prediction loop uses a broad `try...except` block, which can hide specific errors. It would be better to catch specific exceptions (e.g., `PIL.UnidentifiedImageError` for corrupted images) to make debugging easier.

### 5. Add a Detailed Classification Report
While the confusion matrix is useful, a **classification report** from `sklearn.metrics` would provide more granular insights by showing the **precision, recall, and F1-score** for each class.
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=class_names))
```

### 6. Manage Dependencies and Configuration
- **Dependencies**: Instead of a comment, create a `requirements.txt` file to list all project dependencies. This is a standard practice that simplifies environment setup (`pip install -r requirements.txt`).
- **Configuration**: Hardcoded paths and parameters could be moved to a separate configuration file (e.g., `config.yaml`) or passed as command-line arguments using `argparse`. This makes the script more flexible and easier to run with different settings without modifying the code.