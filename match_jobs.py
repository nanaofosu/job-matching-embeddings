import os
import pandas as pd
import openai
from typing import List, Dict
from utils.config import get_env_variable
from utils.data_loader import DataLoader
from utils.embeddings_calculator import EmbeddingsCalculator
from collections import Counter
import re
import spacy
from utils.recommendations import RecommendationsHandler

openai.api_key = get_env_variable('OPENAI_API_KEY')

DEFAULT_EMBEDDING_SIZE = get_env_variable('DEFAULT_EMBEDDING_SIZE', 1536, int)
MAX_RECOMMENDATIONS = get_env_variable('MAX_RECOMMENDATIONS', 5, int)

# Configuration dictionary for keyword categories
KEYWORD_CATEGORIES = {
    "Technical Skills": ["drupal", "html", "web", "data", "python", "javascript", "sql", "css"],
    "Experience": ["project", "management", "leadership", "education", "development", "implementation"],
    "Soft Skills and Culture": ["professional", "excellence", "diverse", "support", "communication", "teamwork"]
}

nlp = spacy.load("en_core_web_sm")  # Load a pre-trained NER model

class JobMatcher:
    def __init__(self):
        self.data_loader = DataLoader()
        self.embeddings_calculator = EmbeddingsCalculator()
        self.recommendations_handler = RecommendationsHandler()
        self.job_listings_df = None
        self.job_descriptions = None
        self.resume_text = None
        self.job_embeddings = None
        self.resume_embedding = None
        self.recommendation_reasons = None
        self.recommendation_summaries = None

    # Load job listings data from CSV file using DataLoader
    def load_data(self):
        self.job_listings_df = self.data_loader.load_csv('data/job_listings.csv')

    # Load resume text from file
    def load_resume(self):
        self.resume_text = self.data_loader.load_text('data/resume.txt')

    # Get job description embedding with error handling
    def get_embedding_with_error_handling(self, description: str, title: str) -> List[float]:
        if not description.strip():  # Check if the description is empty
            print(f"Warning: Empty description encountered for title: {title}.")
            return [0] * DEFAULT_EMBEDDING_SIZE

        try:
            # Combine title and description for embedding generation (consider weighting)
            combined_text = f"{title} {description}"  # Simple concatenation, explore weighting strategies
            embedding = self.embeddings_calculator.get_embeddings(combined_text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for description: {description} (title: {title}). Error: {e}")
            return [0] * DEFAULT_EMBEDDING_SIZE


    def extract_keywords(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)
        doc = nlp(text)  # Parse the text with spaCy
        keywords = []
        for entity in doc.ents:
            if entity.label_ in ("ORG", "SKILL", "XP"):  # Customize labels for skills and experience
                keywords.append(entity.text)
        return keywords

    # Calculate similarities between resume and job descriptions
    def calculate_similarities(self):
        self.job_embeddings = [self.get_embedding_with_error_handling(description, title) for title, description in zip(self.job_listings_df['title'], self.job_descriptions)]

        self.resume_embedding = self.embeddings_calculator.get_embeddings(self.resume_text)

        if self.resume_embedding is None:
            raise ValueError("Failed to get embedding for resume. Please check the resume text and try again.")

        self.similarities = [self.embeddings_calculator.calculate_similarity(self.resume_embedding, job_emb) for job_emb in self.job_embeddings]

    # Rank the job listings based on similarity
    def rank_jobs(self):
        self.job_listings_df['similarity'] = self.similarities
        self.ranked_jobs_df = self.job_listings_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)



    def generate_recommendation_reasons(self):
        resume_skills = self.extract_keywords(self.resume_text)
        recommendation_reasons = []

        for description in self.job_descriptions:
            job_skills = self.extract_keywords(description)
            common_skills = set(resume_skills) & set(job_skills)
            reasons = ', '.join(common_skills)
            recommendation_reasons.append(reasons if reasons else "General match based on job description and resume similarity.")

        self.recommendation_reasons = recommendation_reasons

    # Generate summary based on recommendation reasons
    def generate_summary(self, reasons: str) -> str:
        category_matches = {category: [] for category in KEYWORD_CATEGORIES}

        for word in reasons.split(', '):
            for category, keywords in KEYWORD_CATEGORIES.items():
                if word in keywords:
                    category_matches[category].append(word)

        summary = "Summary\n"
        # summary += "The keywords indicate that the job and your resume have a strong overlap in several important areas:\n\n"
        summary += "The job description mentions skills like " + ", ".join(reasons.split(', ')) + " which directly match your resume.\n"


        for category, matches in category_matches.items():
            if matches:
                summary += f"{category}: " + ", ".join(matches) + ".\n"

        summary += "This strong overlap suggests that you are a good fit for the job based on both your technical and soft skills, as well as your experience and professional background.\n"

        return summary

    # Get top recommended jobs
    def get_recommendations(self):
        recommendations = self.ranked_jobs_df.head(MAX_RECOMMENDATIONS)
        return recommendations

    # Perform job matching process
    def match_jobs(self):
        self.load_data()
        self.job_listings_df['description'] = self.job_listings_df['description'].fillna('')  # Ensure no NaN values
        self.job_descriptions = self.job_listings_df['description'].tolist()
        self.load_resume()
        self.calculate_similarities()
        self.rank_jobs()
        self.generate_recommendation_reasons()
        self.recommendations = self.get_recommendations()

        summaries = [self.generate_summary(reasons) for reasons in self.recommendation_reasons]
        self.recommendation_summaries = summaries

        self.recommendations_handler.display_recommendations(self.recommendations, self.recommendation_reasons, self.recommendation_summaries)
        self.recommendations_handler.write_recommendations_to_md(self.recommendations, self.recommendation_reasons, self.recommendation_summaries)

# Entry point of the program
if __name__ == "__main__":
    job_matcher = JobMatcher()
    job_matcher.match_jobs()
