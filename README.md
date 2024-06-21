
# Job Matching Using Vector Embeddings
## A dating app for your resume!

This project utilizes vector embeddings to match job listings with a resume, enabling you to find the best job fit based on the semantic meaning of the text. It leverages the OpenAI API to generate embeddings and employs cosine similarity to rank job listings.

## Prerequisites

- Python 3.6+
- OpenAI API Key

## Features

- Load job listings data from a CSV file
- Load resume text from a file
- Calculate embeddings for job descriptions and resume using OpenAI
- Calculate similarity between job descriptions and resume
- Display and write job recommendations to a Markdown file


## Installation

1. **Clone the Repository**

    ```bash
    git clone git@github.com:nanaofosu/job-matching-embeddings.git
    cd job-matching-embeddings
    ```

2. **Install Dependencies**

    Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set the following environment variables:**

- OPENAI_API_KEY=your-open-ai-key:  Your OpenAI API key
- DEFAULT_EMBEDDING_SIZE: The size of the embeddings generated (default: 1536)
- MAX_RECOMMENDATIONS: The maximum number of job recommendations to display (default: 5)
- BATCH_SIZE: This sets the batch size (future implemntation)
- CACHE_FILE: This is a file that is stored locally and used as a poor mans cache. (it works)

You can set these variables in a .env file at the root of the project, and they will be loaded automatically when you run the application.


## Usage

### Prepare Data

Prepare your job listings and resume text. Add the job descriptions and resume text in the `data/job_listings.csv` and `data/resume.txt` files, respectively. Ensure that your `job_listings.csv` has the necessary fields, particularly the `description` field.

Craft your perfect resume - or just copy and paste from the internet, we don't judge.  But seriously, update your resume for better results.

Example `job_listings.csv` structure that I use:

```csv
site,job_url,job_url_direct,title,company,location,job_type,date_posted,interval,min_amount,max_amount,currency,is_remote,emails,description,company_url,company_url_direct,company_addresses,company_industry,company_num_employees,company_revenue,company_description,logo_photo_url,banner_photo_url,ceo_name,ceo_photo_url
ExampleSite,https://example.com/job1,https://example.com/job1-direct,Software Developer,Tech Company,San Francisco,Full-time,2024-06-15,Monthly,7000,10000,USD,True,example@example.com,"Job description for software developer...",https://example.com,https://example.com-direct,"123 Main St, San Francisco, CA","Technology",500,1000000000,"Leading tech company in Silicon Valley",https://example.com/logo.png,https://example.com/banner.png,Jane Doe,https://example.com/ceo.jpg
```

### Run the Script

Execute the script to get job recommendations:

```bash
python match_jobs.py
```
### Modules
- `config.py`: Handles the loading and setting of environment variables
- `data_loader.py`: Contains the DataLoader class for loading data from files
- `embeddings_calculator.py`: Contains the EmbeddingsCalculator class for calculating embeddings using the GPT-3 model
- `recommendations.py`: Contains the RecommendationsHandler class for displaying and writing job recommendations
- `match_jobs.py`: Contains the JobMatcher class for matching jobs to a resume

### View Recommendations

The script will output the top 5 job recommendations based on the similarity to your resume.

## Contributing

Contributions are welcome! Please open an issue if you encounter a bug, or a pull request if you have an improvement to suggest.

## License
This project is licensed under the MIT License.
