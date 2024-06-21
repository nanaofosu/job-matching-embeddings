from dotenv import load_dotenv
import openai
import re
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import get_env_variable


class EmbeddingsCalculator:
    def __init__(self):
        # Define CACHE_FILE variable
        self.cache_file = get_env_variable("CACHE_FILE", default_value="cache.json")

        # Load cache from file if it exists
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}  # Initialize empty cache

    def save_cache(self):
        """Save the current cache to a file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def clean_text(self, text):
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

    def get_embeddings(self, text):
        """
        This function takes a text input, cleans it using the clean_text function,
        and then retrieves the embedding for the cleaned text using the OpenAI API.
        It uses a cache to avoid redundant API calls.
        It returns the embedding if successful, or None if there was an error.
        """
        cleaned_text = self.clean_text(text)
        if cleaned_text in self.cache:
            return self.cache[cleaned_text]

        try:
            response = openai.Embedding.create(input=[cleaned_text], model="text-embedding-ada-002")
            embedding = response['data'][0]['embedding']
            self.cache[cleaned_text] = embedding
            self.save_cache()  # Save the updated cache to the file
            return embedding
        except openai.error.InvalidRequestError as e:
            print(f"Failed to get embedding for text: {cleaned_text}")
            print(f"Error: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """
        This function takes two embeddings and calculates the cosine similarity between them.
        It returns the similarity score as a float value.
        """
        return cosine_similarity([embedding1], [embedding2])[0][0]