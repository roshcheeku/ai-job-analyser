import requests
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# API Configuration  
GOOGLE_API_KEY = "AIzaSyDd2Njm34U_vU_uYOPqFjI0S_yrUOMLhyo"  
CUSTOM_SEARCH_ENGINE_ID = "3741929af179a4b17"
JOOBLE_API_URL = "https://jooble.org/api/3f399c3d-2800-474c-bdba-4195187347eb"

# Function to fetch job postings from Google Custom Search
def fetch_google_jobs(queries=None):
    if queries is None:
        queries = [
            "software engineer jobs",
            "python developer jobs",
            "remote software jobs",
            "entry level programming jobs"
        ]

    all_jobs = []
    for query in queries:
        try:
            params = {
                'key': GOOGLE_API_KEY,
                'cx': CUSTOM_SEARCH_ENGINE_ID,
                'q': query,
                'num': 10  # Limit to 10 results per query
            }
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            search_results = response.json()

            jobs = [{
                "title": item.get("title", "No Title"),
                "description": item.get("snippet", "No Description"),
                "link": item.get("link", ""),
                "source": "Google Search"
            } for item in search_results.get("items", [])]
            
            all_jobs.extend(jobs)
        
        except requests.RequestException as e:
            print(f"Error fetching jobs for query '{query}': {e}")
    
    return all_jobs

# Function to fetch job postings from Jooble API
def fetch_jooble_jobs(queries=None):
    if queries is None:
        queries = [
            "software engineer",
            "python developer",
            "remote software jobs",
            "entry level programming"
        ]

    all_jobs = []
    for query in queries:
        try:
            payload = {
                'keywords': query,
                'location': '',  # Empty location for global search or set to a specific location
                'page': '1'
            }

            response = requests.post(JOOBLE_API_URL, json=payload)
            response.raise_for_status()
            jooble_results = response.json()

            jobs = [{
                "title": job.get("title", "No Title"),
                "description": job.get("snippet", "No Description"),
                "link": job.get("link", ""),
                "source": "Jooble"
            } for job in jooble_results.get("jobs", [])]

            all_jobs.extend(jobs)

        except requests.RequestException as e:
            print(f"Error fetching jobs from Jooble for query '{query}': {e}")

    return all_jobs

# Extract skills from job description using NLP
def extract_required_skills(description):
    # Process the job description using spaCy
    doc = nlp(description.lower())
    # Extracting potential skills by filtering for nouns, proper nouns, and verbs (simple approach)
    required_skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]
    return required_skills

# Enhanced Job Recommendation Function with Skill Gap Analysis
def recommend_jobs(user_skills, job_data):
    recommended_jobs = []
    skills = [skill.strip().lower() for skill in user_skills.split(',')]
    
    for job in job_data:
        job_description = job.get("description", "").lower()
        job_title = job.get("title", "").lower()
        
        # Extract the required skills from job description
        job_required_skills = extract_required_skills(job_description)
        
        # Match user's skills with required skills
        skill_matches = [skill for skill in skills if skill in job_required_skills]
        skill_gaps = [skill for skill in job_required_skills if skill not in skills]
        
        if skill_matches:
            job['skill_match_count'] = len(skill_matches)
            job['skill_matches'] = skill_matches
            job['skill_gaps'] = skill_gaps
            recommended_jobs.append(job)
    
    # Sort jobs by the number of skill matches (desc)
    recommended_jobs.sort(key=lambda x: x.get('skill_match_count', 0), reverse=True)
    
    return recommended_jobs

# Main execution
def main():
    try:
        # Fetch job data from Google and Jooble
        google_jobs = fetch_google_jobs()
        jooble_jobs = fetch_jooble_jobs()
        
        # Combine job data from both sources
        job_data = google_jobs + jooble_jobs
        
        print(f"Total jobs fetched: {len(job_data)}")
        
        # Get user skills
        user_skills = input("Enter your skills (comma-separated): ").strip()
        
        # Get recommended jobs
        recommended_jobs = recommend_jobs(user_skills, job_data)
        
        print("\nRecommended Jobs:")
        for job in recommended_jobs[:10]:  # Limit to top 10 recommendations
            print(f"Title: {job.get('title')}")
            print(f"Description: {job.get('description')[:300]}...")
            print(f"Skill Matches: {job.get('skill_match_count', 0)}")
            print(f"Matched Skills: {', '.join(job.get('skill_matches', []))}")
            print(f"Skill Gaps: {', '.join(job.get('skill_gaps', []))}")
            print(f"Source: {job.get('source')}")
            print(f"Link: {job.get('link')}")
            print("---")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()
