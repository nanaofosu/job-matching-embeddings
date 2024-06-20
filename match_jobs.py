import os
import pandas as pd
from utils.embeddings import get_embeddings, calculate_similarity
from dotenv import load_dotenv
import openai
from typing import List

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key (assuming it's already set in .env)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load configuration from .env
DEFAULT_EMBEDDING_SIZE = int(os.getenv('DEFAULT_EMBEDDING_SIZE', 1536))
MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', 5))

def load_data() -> pd.DataFrame:
    try:
        job_listings_df = pd.read_csv('data/job_listings.csv')
        return job_listings_df
    except FileNotFoundError as e:
        raise FileNotFoundError("Job listings CSV file not found. Ensure 'data/job_listings.csv' exists.") from e

def load_resume() -> str:
    try:
        with open('data/resume.txt', 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError("Resume text file not found. Ensure 'data/resume.txt' exists.") from e

def get_embedding_with_error_handling(description: str) -> List[float]:
    try:
        embedding = get_embeddings(description)
        return embedding
    except Exception as e:
        print(f"Error generating embedding for description: {description}. Error: {e}")
        return [0] * DEFAULT_EMBEDDING_SIZE

def main():
    # Load data
    job_listings_df = load_data()
    job_descriptions = job_listings_df['description'].tolist()
    resume_text = load_resume()

    # Get embeddings for job descriptions
    job_embeddings = [get_embedding_with_error_handling(description) for description in job_descriptions]

    # Get embedding for the resume
    resume_embedding = get_embeddings(resume_text)

    # Ensure resume_embedding is valid
    if resume_embedding is None:
        raise ValueError("Failed to get embedding for resume. Please check the resume text and try again.")

    # Calculate similarities
    similarities = [calculate_similarity(resume_embedding, job_emb) for job_emb in job_embeddings]

    # Add similarity scores to the dataframe
    job_listings_df['similarity'] = similarities

    # Sort the dataframe based on similarity scores
    ranked_jobs_df = job_listings_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # Get top recommendations based on MAX_RECOMMENDATIONS
    recommendations = ranked_jobs_df.head(MAX_RECOMMENDATIONS)

    # Display the recommendations
    for idx, row in recommendations.iterrows():
        print(f"Recommendation {idx+1}:")
        print(f"Title: {row['title']}")
        print(f"Company: {row['company']}")
        print(f"Location: {row['location']}")
        print(f"Job Type: {row['job_type']}")
        print(f"Date Posted: {row['date_posted']}")
        print(f"Company URL: {row['company_url']}")
        print(f"Job URL: {row['job_url']}")
        print(f"Job Direct URL: {row['job_url_direct']}")
        print()

if __name__ == "__main__":
    main()
