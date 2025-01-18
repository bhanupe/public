import os
import csv

def clean_review(review):
    """
    Clean the review text to ensure it does not interfere with the delimiter.
    - Replaces stray tabs with spaces.
    - Removes extra spaces and escape characters.
    """
    review = review.strip()  # Remove leading and trailing whitespace
    review = review.replace("\t", " ")  # Replace tabs with spaces
    review = review.replace("\n", " ")  # Replace newlines with spaces
    review = review.replace("\\", "")  # Remove backslashes
    review = review.replace('"', "'")  # Replace double quotes with single quotes
    review = " ".join(review.split())  # Collapse multiple spaces into one
    return review

def process_and_fix_reviews(input_path, output_csv):
    """
    Process reviews from train/pos and train/neg directories, clean them,
    and create a single clean CSV file.

    Args:
        input_path (str): Path to the ⁠ train ⁠ directory containing ⁠ pos ⁠ and ⁠ neg ⁠ folders.
        output_csv (str): Output CSV file path.
    """
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Review", "Sentiment"])  # Header row

        # Loop through 'pos' and 'neg' folders
        for folder, sentiment in [("pos", "positive"), ("neg", "negative")]:
            folder_path = os.path.join(input_path, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        review = file.read()
                        cleaned_review = clean_review(review)

                        # Ensure valid row structure
                        if not cleaned_review:
                            print(f"Skipping empty review in file: {file_path}")
                            continue

                        try:
                            writer.writerow([cleaned_review, sentiment])
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")

    print(f"Cleaned and processed CSV file saved as {output_csv}")

if __name__ == "__main__":
    input_path = "aclImdb/train"  # Adjust this path to your dataset's location
    output_csv = "movie_reviews.csv"  # Final output file name

    # Process and clean the reviews, generating the final single CSV file
    process_and_fix_reviews(input_path, output_csv)
