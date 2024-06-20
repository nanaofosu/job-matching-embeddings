
# Job Matching Using Vector Embeddings

This project utilizes vector embeddings to match job listings with a resume, enabling you to find the best job fit based on the semantic meaning of the text. It leverages the OpenAI API to generate embeddings and employs cosine similarity to rank job listings.

## Prerequisites

- Python 3.6+
- OpenAI API Key

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

3. **Set Up OpenAI API Key**

    Obtain your OpenAI API key from OpenAI. Set the API key as an environment variable or directly in the script.

## Usage

### Prepare Data

Prepare your job listings and resume text. Add the job descriptions and resume text in the `data/job_listings.csv` and `data/resume.txt` files, respectively. Ensure that your `job_listings.csv` has the necessary fields, particularly the `description` field.

Example `job_listings.csv` structure:

```csv
site,job_url,job_url_direct,title,company,location,job_type,date_posted,interval,min_amount,max_amount,currency,is_remote,emails,description,company_url,company_url_direct,company_addresses,company_industry,company_num_employees,company_revenue,company_description,logo_photo_url,banner_photo_url,ceo_name,ceo_photo_url
ExampleSite,https://example.com/job1,https://example.com/job1-direct,Software Developer,Tech Company,San Francisco,Full-time,2024-06-15,Monthly,7000,10000,USD,True,example@example.com,"Job description for software developer...",https://example.com,https://example.com-direct,"123 Main St, San Francisco, CA","Technology",500,1000000000,"Leading tech company in Silicon Valley",https://example.com/logo.png,https://example.com/banner.png,Jane Doe,https://example.com/ceo.jpg
```

### Run the Script

Execute the script to get job recommendations:

```bash
python match_jobs.py
```

### View Recommendations

The script will output the top 5 job recommendations based on the similarity to your resume.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
