# Part 1

## Abstract Summary: Historical Structures Image Classification Project

This project implements a deep learning pipeline to classify images of historical architectural features. The process begins with data loading and exploratory data analysis (EDA) to visualize image samples and analyze class distribution. A transfer learning approach is adopted, using a pre-trained MobileNetV2 model with a custom classification head. The model is trained on an 80/20 training/validation split, with early stopping to prevent overfitting. Key performance metrics, such as accuracy and loss, are visualized, and the final trained model is saved. Evaluation is performed on a separate test set, with results presented in a confusion matrix.

The review also identifies several areas for potential improvement. Key suggestions include enhancing data augmentation to improve model generalization, implementing systematic hyperparameter tuning to optimize performance, and refactoring the code into a more modular structure for better readability and reusability. Additional recommendations involve improving error handling, generating more detailed classification reports (including precision, recall, and F1-scores), and managing dependencies and configurations more formally through a `requirements.txt` file and a dedicated configuration file.

# Part 2

## Abstract Summary: Recommendation Engine
This script develops a tourist recommendation engine using a collaborative filtering approach with the `surprise` library.

Here is a summary of its key operations:

1.  **Data Loading**: It begins by loading user ratings for various tourist places from `data/part2/tourism_rating.csv` into a pandas DataFrame. This data, containing user IDs, place IDs, and ratings, is then loaded into a `surprise` Dataset format, which is optimized for recommendation algorithms.

2.  **Model Training**: The script employs the Singular Value Decomposition (SVD) algorithm, a popular matrix factorization technique for collaborative filtering. It trains the SVD model on the entire set of user ratings to learn latent preference patterns.

3.  **Recommendation Generation**: The core logic is encapsulated in the `get_recommendations` function. For a given user, this function:
    *   Identifies all the places the user has not yet rated.
    *   Uses the trained SVD model to predict a rating for each of these unrated places.
    *   Sorts the places by their predicted rating in descending order.
    *   Returns the top N places as recommendations, along with their estimated ratings.

4.  **Outputting Recommendations**:
    *   The script loads detailed information about the tourist attractions from an Excel file (`data/part2/tourism_with_id.xlsx`).
    *   It reads a list of user IDs from `data/part2/user.csv`.
    *   For each user, it generates the top 5 recommendations.
    *   Finally, it prints each recommended place's ID and estimated rating, then looks up and displays detailed information for that place, such as its name, description, city, and price.