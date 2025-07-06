# Develop a simple recommendation engine to suggest places of interest to tourists.
# Use collaborative filtering approach with the Surprise library.

import pandas as pd
from surprise import Dataset, Reader, SVD

ratings_df = pd.read_csv('data/part2/tourism_rating.csv')

# Load data into Surprise's Dataset format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

# 3. Train a collaborative filtering model (SVD)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Generate recommendations for a user
def get_recommendations(user_id, max_recommendations=5):
    # Get a list of all item IDs
    all_item_ids = ratings_df['Place_Id'].unique()

    # Get items the user has already rated
    rated_items = ratings_df[ratings_df['User_Id'] == user_id]['Place_Id'].unique()

    # Predict ratings for unrated items
    unrated_items = [item for item in all_item_ids if item not in rated_items]
    predictions = [algo.predict(user_id, item_id) for item_id in unrated_items]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_mr = predictions[:max_recommendations]

    return [(pred.iid, pred.est) for pred in top_mr]


# Load tourism data
tourism_df = pd.read_excel('data/part2/tourism_with_id.xlsx')

# Example: Get recommendations for users from users.csv
users_df = pd.read_csv('data/part2/user.csv')
for user_id_to_recommend in users_df['User_Id']:
    recommendations = get_recommendations(user_id_to_recommend)

    print(f"\nTop 5 Recommendations for User {user_id_to_recommend} ---")
    for item, estimated_rating in recommendations:
        print(f"\tRecommended Category: {item}, Estimated Rating: {estimated_rating:.2f}")

        try:
            # Find places in the recommended category with a similar rating
            similar_places = tourism_df[
                (tourism_df['Place_Id'] == item) &
                (tourism_df['Rating'] >= estimated_rating * 0.9) &
                (tourism_df['Rating'] <= estimated_rating * 1.1)
            ]

            print(f"\tPlaces in category '{item}' with similar rating ---")
            if not similar_places.empty:
                for _, row in similar_places.iterrows():
                    print(f"\tPlace: {row['Place_Id']}\n\tPlace: {row['Place_Name']}\n\tDescription: {row['Description']}\n\tCity: {row['City']}\n\tPrice: {row['Price']}\n\tRating: {row['Rating']:.2f}\n")
            else:
                print("\t\tNo similar places found\n")
        except KeyError as e:
            print(f"\tAn error occurred while finding similar places. A required column is missing: {e}")