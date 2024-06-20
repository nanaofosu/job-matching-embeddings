import openai
import re
from sklearn.metrics.pairwise import cosine_similarity

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

def get_embedding(text):
    """
    This function takes a text input, cleans it using the clean_text function,
    and then retrieves the embedding for the cleaned text using the OpenAI API.
    It returns the embedding if successful, or None if there was an error.
    """
    text = clean_text(text)
    try:
        response = openai.Embedding.create(input=[text], model="text-embedding-3-small")
        return response['data'][0]['embedding']
    except openai.error.InvalidRequestError as e:
        print(f"Failed to get embedding for text: {text}")
        print(f"Error: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """
    This function takes two embeddings and calculates the cosine similarity between them.
    It returns the similarity score as a float value.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]
