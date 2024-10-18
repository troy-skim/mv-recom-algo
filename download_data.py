import os
import requests

# Create the data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# URL to the dataset (replace with the actual dataset link)
url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
# URL to the smaller dataset for development process
# url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(url)

# Save the dataset in the data/raw folder
with open("data/raw/ml-latest.zip", "wb") as f:
    f.write(response.content)

print("Dataset downloaded and saved to 'data/raw/ml-latest.zip'")