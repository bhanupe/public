# Gemini API Integration and Review Sentiment Analysis

This project demonstrates integrating with the Gemini Developer API, classifying reviews for sentiment analysis, and managing API interaction efficiently.

---

## Getting Started

### 1. Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or later

---

### 2. Setting Up the Gemini API
To use the Gemini API in this project:

1. **Create a Gemini API Key**:
   - Sign up or log in to the [Gemini Developer Portal](https://developers.gemini.com).
   - Navigate to the API Keys section.
   - Create a new API key and note down the **API Key** and **API Secret** securely.

2. **Store the API Key**:
   - Create a `config.txt` file in the `utilities/api_interaction` folder.
   - Add the following line to the file:
     ```
     API_KEY=your_api_key_here
     ```
Running Python code to 

Reads reviews from a file (default: classify_reviews.csv).
Constructs a question dynamically to analyze the sentiment of each review.
Sends the question to the Gemini AI API for processing.
Logs the question, review, and API response to a file.
Output results to both the terminal and a specified output file (output.txt).

the output and classify_reviews.csv files will be uploaded to Drive 

---

### 3. Cloning the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/bhanupe/public.git
cd public

# Create a new branch for your changes (replace "yt-upw-4" with your branch name)
git checkout -b yt-upw-4 

# Create the config.txt file in the utilities/api_interaction directory 
# Add your Gemini API key to the config.txt file: 
# API_KEY=your_api_key_here 
# Stage the gemini_api.py and config.txt files 
git add utilities/api_interaction/gemini_api.py  
# Commit your changes 
git commit -m "yt-upw-4 Add Gemini API integration and configuration" 
# Push the changes to the remote repository 
git push origin yt-upw-4

