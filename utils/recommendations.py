from datetime import datetime

class RecommendationsHandler:
    @staticmethod
    def display_recommendations(recommendations, recommendation_reasons, recommendation_summaries):
        for idx, row in recommendations.iterrows():
            print(f"Recommendation {idx+1}:")
            print(f"Title: {row['title']}")
            print(f"Company: {row['company']}")
            print(f"Location: {row['location']}")
            print(f"Job Type: {row['job_type']}")
            print(f"Date Posted: {row['date_posted']}")
            print(f"Similarity Score: {row['similarity']:.2f}")
            print(f"Description: {row['description'][:200]}...")  # Display a snippet of the description
            print(f"Company URL: {row['company_url']}")
            print(f"Job URL: {row['job_url']}")
            print(f"Job Direct URL: {row['job_url_direct']}")
            print(f"Recommendation Reason: {recommendation_reasons[idx]}")
            print(f"Summary: {recommendation_summaries[idx]}")
            print()

    @staticmethod
    def write_recommendations_to_md(recommendations, recommendation_reasons, recommendation_summaries):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"recommendations-{timestamp}.md"

        with open(filename, 'w') as f:
            f.write(f"# Job Recommendations ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
            for idx, row in recommendations.iterrows():
                f.write(f"## Recommendation {idx+1}:\n")
                f.write(f"**Title:** {row['title']}\n\n")
                f.write(f"**Company:** {row['company']}\n\n")
                f.write(f"**Location:** {row['location']}\n\n")
                f.write(f"**Job Type:** {row['job_type']}\n\n")
                f.write(f"**Date Posted:** {row['date_posted']}\n\n")
                f.write(f"**Similarity Score:** {row['similarity']:.2f}\n\n")
                f.write(f"**Description:** {row['description'][:200]}...\n\n")  # Display a snippet of the description
                f.write(f"**Company URL:** {row['company_url']}\n\n")
                f.write(f"**Job URL:** {row['job_url']}\n\n")
                f.write(f"**Job Direct URL:** {row['job_url_direct']}\n\n")
                f.write(f"**Recommendation Reason:** {recommendation_reasons[idx]}\n\n")
                f.write(f"**Summary:** {recommendation_summaries[idx]}\n\n")
                f.write("---\n\n")
        print(f"Recommendations written to {filename}")