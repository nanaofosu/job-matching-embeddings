import openai
import re
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# Define CACHE_FILE variable
CACHE_FILE = os.getenv("CACHE_FILE", "cache.json")

# Load cache from file if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
else:
    cache = {}  # Initialize empty cache

def save_cache():
    """Save the current cache to a file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def clean_text(text):
    """
    This function takes a text input and performs several cleaning operations on it.
    It removes email addresses, replaces multiple spaces with a single space,
    removes non-ASCII characters, removes problematic characters (e.g., asterisks),
    and removes leading and trailing whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'[*]', '', text)  # Remove problematic characters (e.g., asterisks)
    text = text.strip()  # Remove leading and trailing whitespace

    return text

def get_embeddings(text):
    """
    This function takes a text input, cleans it using the clean_text function,
    and then retrieves the embedding for the cleaned text using the OpenAI API.
    It uses a cache to avoid redundant API calls.
    It returns the embedding if successful, or None if there was an error.
    """
    cleaned_text = clean_text(text)
    if cleaned_text in cache:
        return cache[cleaned_text]

    try:
        response = openai.Embedding.create(input=[cleaned_text], model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        cache[cleaned_text] = embedding
        save_cache()  # Save the updated cache to the file
        return embedding
    except openai.error.InvalidRequestError as e:
        print(f"Failed to get embedding for text: {cleaned_text}")
        print(f"Error: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """
    This function takes two embeddings and calculates the cosine similarity between them.
    It returns the similarity score as a float value.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Example usage
if __name__ == "__main__":
    texts = [
        "Example text 1",
        "Example text 2",
        "Another example text",
        "And another one!"
    ]
    embeddings = {text: get_embeddings(text) for text in texts}
    for text, embedding in embeddings.items():
        print(f"Text: {text}\nEmbedding: {embedding}")

    # Example similarity calculation
    if len(embeddings) >= 2:
        texts_list = list(embeddings.keys())
        sim_score = calculate_similarity(embeddings[texts_list[0]], embeddings[texts_list[1]])
        print(f"Similarity between '{texts_list[0]}' and '{texts_list[1]}': {sim_score}")
