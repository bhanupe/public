import os
import json
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from analytical_processors.historcal_structures.predictions import predict_image_category
from analytical_processors.utilities.file_utils import unzip_file


############################################## EXTRACT DATA FOR TRAINING ###############################################
# copy dataset_hist_structures.zip to data/part1 folder
# copy Part 2 to data/part2 folder
home_directory = 'data'
part1_path = f'{home_directory}/part1'

if not os.path.exists(f'{part1_path}/dataset_hist_structures'):
    # Example usage:
    # Replace 'your_zip_file.zip' with the name of your zip file
    # Replace 'extracted_content' with the directory you want to extract to
    zip_file_name = f'{part1_path}/dataset_hist_structures.zip'
    extraction_directory = f'{part1_path}'

    # Create the extraction directory if it doesn't exist
    os.makedirs(extraction_directory, exist_ok=True)

    unzip_file(zip_file_name, extraction_directory)
else:
    print("zip file already extracted")

################################################ SET STAGE FOR TRAINING ################################################
# Replace with the path to your main image folder
    image_dir = f'{part1_path}/dataset_hist_structures/Stuctures_Dataset'
    image_height = 150
    image_width = 150
    batch_size = 32
    epochs = 10

    # Use ImageDataGenerator to load images and labels from folders
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2)  # Rescale pixel values and split for validation

    train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',  # Use 'categorical' for multiple classes
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(train_generator.num_classes, activation='softmax')
        # Output layer with number of classes and softmax activation
    ])
if not os.path.exists(f'{part1_path}/my_image_classification_model.keras'):
    ############################################# BUILD A SIMPLE CNN MODEL #############################################


    ################################################ COMPILE THE MODEL #################################################
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ################################################# TRAIN THE MODEL ##################################################
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    ############################################# SAVE THE TRAINED MODEL ###############################################
    # Alternatively, you can save in other formats:
    # model.save('my_image_classification_model_savedmodel') # Saves in the SavedModel format
    # model.save('my_image_classification_model.h5') # Saves in the HDF5 format
    model.save(f'{part1_path}/my_image_classification_model.keras')  # Saves in the native Keras format
    print("Model saved successfully.")

    ######################################## SAVE THE TRAINING HISTORY MODEL ###########################################
    with open(f'{part1_path}/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved successfully.")

    ############################################ SAVE THE CLASS LABELS #################################################
    class_labels = list(train_generator.class_indices.keys())
    with open(f'{part1_path}/class_labels.json', 'w') as f:
        json.dump(class_labels, f)
    print("Class labels saved successfully.")
else:
    print("model training already complete")

############################################ PREDICT CLASS OF AN IMAGE #################################################
# Replace with the path to your new image
# We find that the prediciton is very random.
new_image_path = f'{part1_path}/dataset_hist_structures/Dataset_test/Dataset_test_original_1478/altar/00947185-77b4-4ecf-85ed-bf139c7d5a1d.jpg'
class_labels = list(train_generator.class_indices.keys()) # Get the class labels from the generator

predicted_category, predictions = predict_image_category(model, new_image_path, image_height, image_width, class_labels)

print(f"The new image is predicted to belong to the category: {predicted_category}")
print(f"Prediction probabilities: {predictions}")