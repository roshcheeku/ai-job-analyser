import requests
import spacy
import time
import tkinter as tk
from tkinter import simpledialog

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# API Configuration  
GOOGLE_API_KEY = "api-key"  
CUSTOM_SEARCH_ENGINE_ID = "engine-id"
JOOBLE_API_URL = "jobbleapi"

MAX_RETRIES = 3  # Maximum number of retries for failed API requests

# Function to fetch job postings from Google Custom Search
def fetch_google_jobs(queries=None, job_role=None, location=None):
    if queries is None:
        queries = [
            f"{job_role} jobs",
            "software engineer jobs",
            "python developer jobs",
            "remote software jobs"
        ]
    
    if location:
        queries = [query + f" in {location}" for query in queries]

    all_jobs = []
    for query in queries:
        retries = 0
        while retries < MAX_RETRIES:
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

                jobs = [({
                    "title": item.get("title", "No Title"),
                    "description": item.get("snippet", "No Description"),
                    "link": item.get("link", ""),
                    "source": "Google Search",
                    "salary": item.get("formattedValue", ""),
                    "company": item.get("displayLink", ""),
                    "location": item.get("pagemap", {}).get("postalAddress", [{}])[0].get("addressLocality", "")
                }) for item in search_results.get("items", [])]

                all_jobs.extend(jobs)
                break  # If successful, exit retry loop

            except requests.RequestException as e:
                print(f"Error fetching jobs for query '{query}': {e}")
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(2)  # Wait before retrying

    return all_jobs


# Function to fetch job postings from Jooble API
def fetch_jooble_jobs(queries=None, job_role=None, location=None):
    if queries is None:
        queries = [
            f"{job_role}",
            "software engineer",
            "python developer",
            "remote software jobs"
        ]
    
    all_jobs = []
    for query in queries:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                payload = {
                    'keywords': query,
                    'location': location if location else '',  # Use location if provided
                    'page': '1'
                }

                response = requests.post(JOOBLE_API_URL, json=payload)
                response.raise_for_status()
                jooble_results = response.json()

                jobs = [({
                    "title": job.get("title", "No Title"),
                    "description": job.get("snippet", "No Description"),
                    "link": job.get("link", ""),
                    "source": "Jooble",
                    "salary": job.get("salary", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", "")
                }) for job in jooble_results.get("jobs", [])]

                all_jobs.extend(jobs)
                break  # If successful, exit retry loop

            except requests.RequestException as e:
                print(f"Error fetching jobs from Jooble for query '{query}': {e}")
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(2)  # Wait before retrying

    return all_jobs


# Extract skills from job description using NLP
def extract_required_skills(description):
    # Process the job description using spaCy
    doc = nlp(description.lower())
    # Extracting potential skills by filtering for nouns, proper nouns, and verbs
    required_skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]
    return required_skills


# Enhanced Role Matching Function
def enhance_role_matching(user_role, job_title, job_description):
    # Convert user role and job title/description to lowercase for case-insensitive matching
    user_role_lower = user_role.lower()
    job_title_lower = job_title.lower()
    job_description_lower = job_description.lower()

    # Check if the user role or any synonym of the user role exists in the job title or description
    if user_role_lower in job_title_lower or user_role_lower in job_description_lower:
        return True
    else:
        return False


# Enhanced Job Recommendation Function with Skill Gap Analysis
def recommend_jobs(user_skills, job_data, user_role, min_skill_match=1):
    recommended_jobs = []
    skills = [skill.strip().lower() for skill in user_skills.split(',')]
    
    for job in job_data:
        job_description = job.get("description", "").lower()
        job_title = job.get("title", "").lower()
        
        # Check if the job is relevant to the user's preferred role
        if enhance_role_matching(user_role, job_title, job_description):
            # Extract the required skills from job description
            job_required_skills = extract_required_skills(job_description)
            
            # Match user's skills with required skills
            skill_matches = [skill for skill in skills if skill in job_required_skills]
            skill_gaps = [skill for skill in job_required_skills if skill not in skills]

            # Calculate job relevance score based on the number of matched skills
            job_relevance_score = len(skill_matches)
            
            # Only recommend jobs that meet the minimum skill match threshold
            if job_relevance_score >= min_skill_match:
                # Add skill matches and gaps to the job data
                job['skill_match_count'] = job_relevance_score
                job['skill_matches'] = skill_matches
                job['skill_gaps'] = skill_gaps
                job['relevance_score'] = job_relevance_score
                recommended_jobs.append(job)

    # Sort jobs by the relevance score (desc)
    recommended_jobs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return recommended_jobs


# Function to collect user details using a pop-up window
def collect_user_details():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    name = simpledialog.askstring("User Details", "Name:")
    email = simpledialog.askstring("User Details", "Email:")
    phone = simpledialog.askstring("User Details", "Phone Number:")
    cgpa = simpledialog.askstring("User Details", "CGPA of B.E:")
    skills = simpledialog.askstring("User Details", "Technical Skills (comma-separated):")
    soft_skills = simpledialog.askstring("User Details", "Soft Skills (comma-separated):")
    certifications = simpledialog.askstring("User Details", "Certifications (comma-separated):")
    job_role = simpledialog.askstring("User Details", "Job Role Applied For:")
    experience = simpledialog.askstring("User Details", "Previous Experience (if any):")
    location = simpledialog.askstring("User Details", "Preferred Location (leave blank for global search):")

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "cgpa": cgpa,
        "skills": skills,
        "soft_skills": soft_skills,
        "certifications": certifications,
        "job_role": job_role,
        "experience": experience,
        "location": location
    }


# Main execution
def main():
    try:
        # Collect user details using the pop-up window
        user_details = collect_user_details()
        
        print("\nUser Details Collected:")
        for key, value in user_details.items():
            print(f"{key.capitalize()}: {value}")

        # Fetch job data from Google and Jooble
        google_jobs = fetch_google_jobs(job_role=user_details["job_role"], location=user_details["location"])
        jooble_jobs = fetch_jooble_jobs(job_role=user_details["job_role"], location=user_details["location"])
        
        # Combine job data from both sources
        job_data = google_jobs + jooble_jobs
        
        print(f"\nTotal jobs fetched: {len(job_data)}")
        
        # Get recommended jobs
        recommended_jobs = recommend_jobs(user_details["skills"], job_data, user_details["job_role"], min_skill_match=1)
        
        print("\nRecommended Jobs:")
        for job in recommended_jobs[:10]:  # Limit to top 10 recommendations
            print(f"Title: {job.get('title')}")
            print(f"Description: {job.get('description')[:300]}...")
            print(f"Salary: {job.get('salary', '')}")
            print(f"Company: {job.get('company', '')}")
            print(f"Location: {job.get('location', '')}")
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
