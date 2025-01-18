import os
import csv

def process_reviews(input_path, output_csv):
    """
    Process reviews from train/pos and train/neg directories to create a clean CSV file.

    Args:
        input_path (str): Path to the ⁠ train ⁠ directory containing ⁠ pos ⁠ and ⁠ neg ⁠ folders.
        output_csv (str): Output CSV file path.
    """
    rows = []

    # Loop through 'pos' and 'neg' folders
    for folder, sentiment in [("pos", "positive"), ("neg", "negative")]:
        folder_path = os.path.join(input_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    review = file.read().strip().replace("\n", " ")  # Remove extra newlines
                    rows.append([review, sentiment])

    # Write rows to CSV with tab as the delimiter
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["Review", "Sentiment"])  # Header row
        writer.writerows(rows)

    print(f"CSV file created successfully: {output_csv}")

if __name__ == "__main__":
    input_path = "aclImdb/train"  # Adjust this path to your dataset's location
    output_csv = "movie_reviews.csv"  # Output file name
    process_reviews(input_path, output_csv)
