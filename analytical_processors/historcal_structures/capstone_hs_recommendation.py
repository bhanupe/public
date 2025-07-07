# Develop a simple recommendation engine to suggest places of interest to tourists.
# Use collaborative filtering approach with the Surprise library.

import pandas as pd
from surprise import Dataset, Reader, SVD
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from surprise.model_selection import cross_validate
import os
from datetime import datetime


def get_datetime():
    now = datetime.now()

    # It is also possible to format the output:
    return now.strftime("%Y%m%d%H%M%S")


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


show = True
folder = get_datetime()
create_folder(folder)


def save_show(plot, name):
    plot.savefig(f'{folder}/{name}.png')
    if show:
        plot.show()


warnings.filterwarnings("ignore")

# part 2: Method 1
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

    print(f"\n**Top Recommendations for User {user_id_to_recommend}**")
    for item, estimated_rating in recommendations:
        print(f"- Recommended Category: {item}, Estimated Rating: {estimated_rating:.2f}")

        try:
            # Find places in the recommended category with a similar rating
            similar_places = tourism_df[
                (tourism_df['Place_Id'] == item) &
                (tourism_df['Rating'] >= estimated_rating * 0.9) &
                (tourism_df['Rating'] <= estimated_rating * 1.1)
                ]

            print(f"- Places in category '{item}' with similar rating ---")
            if not similar_places.empty:
                for _, row in similar_places.iterrows():
                    print(
                        f"- Place: {row['Place_Id']}\n- Place: {row['Place_Name']}\n- Description: {row['Description']}\n- City: {row['City']}\n- Price: {row['Price']}\n- Rating: {row['Rating']:.2f}\n")
            else:
                print("- - No similar places found\n")
        except KeyError as e:
            print(f"- An error occurred while finding similar places. A required column is missing: {e}")

# part 2: Method 2
users = pd.read_csv("data/part2/user.csv")
places = pd.read_excel("data/part2/tourism_with_id.xlsx")
ratings = pd.read_csv("data/part2/tourism_rating.csv")

users.drop_duplicates(inplace=True)
places.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

print("sums the NAN values column-wise, giving you the total number of missing values in each column of the DataFrame")
print(users.isnull().sum())
print(places.isnull().sum())
print(ratings.isnull().sum())

sns.histplot(users['Age'], bins=20, kde=True)
plt.title("Age Distribution of Tourists")
save_show(plt, "AgeDistributionOfTourists")

sns.countplot(y='Location', data=users, order=users['Location'].value_counts().index[:10])
plt.title("Top Tourist Origins")
save_show(plt, "TopTouristOrigins")

merged = ratings.merge(users, on='User_Id').merge(places, on='Place_Id')

top_places = merged.groupby('Place_Name')['Place_Ratings'].mean().sort_values(ascending=False)
print(top_places.head())

best_cities = merged.groupby('City')['Place_Ratings'].mean().sort_values(ascending=False)
print(best_cities.head())

best_categories = merged.groupby('Category')['Place_Ratings'].mean().sort_values(ascending=False)
print(best_categories.head())

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

place_ids = ratings['Place_Id'].unique()

# user_id = 'U1003'
# recommendations = [(pid, algo.predict(user_id, pid).est) for pid in place_ids]
# recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
#
# print("Top Recommendations for User:", user_id)
# for pid, score in recommendations[:5]:
#     name = places[places['Place_Id'] == pid]['Place_Name'].values[0]
#     print(f"{name} (Predicted Rating: {score:.2f})")

users_df = pd.read_csv('data/part2/user.csv')
for user_id_to_recommend in users_df['User_Id']:
    recommendations = [(pid, algo.predict(user_id_to_recommend, pid).est) for pid in place_ids]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    print("\n**Top 5 Recommendations for User:**", user_id_to_recommend)
    for pid, score in recommendations[:5]:
        name = places[places['Place_Id'] == pid]['Place_Name'].values[0]
        print(f"- {name} (Predicted Rating: {score:.2f})")
