import os
import pandas as pd
# from utils.embeddings import get_embeddings, calculate_similarity
import openai
from typing import List
from utils.config import get_env_variable
from utils.data_loader import DataLoader
from utils.embeddings_calculator import EmbeddingsCalculator

openai.api_key = get_env_variable('OPENAI_API_KEY')

DEFAULT_EMBEDDING_SIZE = get_env_variable('DEFAULT_EMBEDDING_SIZE', 1536, int)
MAX_RECOMMENDATIONS = get_env_variable('MAX_RECOMMENDATIONS', 5, int)

# Define a class to handle job matching
class JobMatcher:
    def __init__(self):
        self.data_loader = DataLoader()
        self.embeddings_calculator = EmbeddingsCalculator()
        self.job_listings_df = None
        self.job_descriptions = None
        self.resume_text = None
        self.job_embeddings = None
        self.resume_embedding = None

    # Load job listings data from CSV file using DataLoader
    def load_data(self):
        self.job_listings_df = self.data_loader.load_csv('data/job_listings.csv')

    # Load resume text from file
    def load_resume(self):
       self.resume_text = self.data_loader.load_text('data/resume.txt')

    # Get job description embedding with error handling
    def get_embedding_with_error_handling(self, description: str) -> List[float]:
        try:
            embedding = self.embeddings_calculator.get_embeddings(description)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for description: {description}. Error: {e}")
            return [0] * DEFAULT_EMBEDDING_SIZE

    # Calculate similarities between resume and job descriptions
    def calculate_similarities(self):
        self.job_embeddings = [self.get_embedding_with_error_handling(description) for description in self.job_descriptions]
        self.resume_embedding = self.embeddings_calculator.get_embeddings(self.resume_text)

        if self.resume_embedding is None:
            raise ValueError("Failed to get embedding for resume. Please check the resume text and try again.")

        self.similarities = [self.embeddings_calculator.calculate_similarity(self.resume_embedding, job_emb) for job_emb in self.job_embeddings]

    # Rank the job listings based on similarity
    def rank_jobs(self):
        self.job_listings_df['similarity'] = self.similarities
        self.ranked_jobs_df = self.job_listings_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # Get top recommended jobs
    def get_recommendations(self):
        recommendations = self.ranked_jobs_df.head(MAX_RECOMMENDATIONS)
        return recommendations

    # Display the recommended jobs
    def display_recommendations(self):
        for idx, row in self.recommendations.iterrows():
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

    # Perform job matching process
    def match_jobs(self):
        self.load_data()
        self.job_descriptions = self.job_listings_df['description'].tolist()
        self.load_resume()
        self.calculate_similarities()
        self.rank_jobs()
        self.recommendations = self.get_recommendations()
        self.display_recommendations()

# Entry point of the program
if __name__ == "__main__":
    job_matcher = JobMatcher()
    job_matcher.match_jobs()
