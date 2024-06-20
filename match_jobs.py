import os
import pandas as pd
from utils.embeddings import get_embedding, calculate_similarity
from dotenv import load_dotenv
import openai

# Load environment variables from the .env file
load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load data
job_listings_df = pd.read_csv('data/job_listings.csv')
job_descriptions = job_listings_df['description'].tolist()

with open('data/resume.txt', 'r') as f:
    resume_text = f.read()

# Get embeddings for job descriptions
job_embeddings = []
for idx, description in enumerate(job_descriptions):
    embedding = get_embedding(description)
    if embedding is not None:
        job_embeddings.append(embedding)
    else:
        print(f"Description at index {idx} failed: {description}")
        job_embeddings.append([0]*1536)  # Assuming the embedding size for the model used is 1536

# Get embedding for the resume
resume_embedding = get_embedding(resume_text)

# Ensure resume_embedding is valid
if resume_embedding is None:
    raise ValueError("Failed to get embedding for resume. Please check the resume text and try again.")

# Calculate similarities
similarities = [calculate_similarity(resume_embedding, job_emb) for job_emb in job_embeddings]

# Add similarity scores to the dataframe
job_listings_df['similarity'] = similarities

# Sort the dataframe based on similarity scores
ranked_jobs_df = job_listings_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

# Get top 5 recommendations
recommendations = ranked_jobs_df.head(5)

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