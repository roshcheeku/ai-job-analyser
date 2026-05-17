import json
import os
import requests
from collections import Counter
from transformers import pipeline
import spacy

# Initialize spaCy for skill extraction
nlp = spacy.load("en_core_web_sm")

# API Configuration  
GOOGLE_API_KEY = ""  
CUSTOM_SEARCH_ENGINE_ID = ""
JOOBLE_API_URL = ""
ADZUNA_APP_ID = ""
ADZUNA_APP_KEY = ""
TEMP_JOB_FILE = "temp_jobs.json"

# Known skills for filtering extracted entities (can be expanded)
KNOWN_SKILLS = {"python", "javascript", "react", "css", "html", "django", "typescript", "graphql"}

# Initialize NER pipeline for skill extraction using transformers
skill_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Helper Functions
def fetch_google_jobs(queries, location=""):
    """Fetch job listings using Google Custom Search API."""
    all_jobs = []
    for query in queries:
        try:
            params = {
                'key': GOOGLE_API_KEY,
                'cx': CUSTOM_SEARCH_ENGINE_ID,
                'q': f"{query} {location}",
                'num': 10
            }
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            search_results = response.json()
            jobs = [
                {
                    "title": item.get("title", "No Title"),
                    "description": item.get("snippet", "No Description"),
                    "link": item.get("link", ""),
                    "source": "Google Search"
                }
                for item in search_results.get("items", [])
            ]
            all_jobs.extend(jobs)
        except requests.RequestException as e:
            print(f"Error fetching jobs for query '{query}': {e}")
    return all_jobs


def fetch_jooble_jobs():
    """Fetch job listings using Jooble API."""
    try:
        response = requests.post(JOOBLE_API_URL, json={"keywords": "software engineer"})
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
        all_jobs = [
            {
                "title": job.get("title", "No Title"),
                "description": job.get("snippet", "No Description"),
                "link": job.get("url", ""),
                "source": "Jooble"
            }
            for job in jobs
        ]
        return all_jobs
    except requests.RequestException as e:
        print(f"Error fetching jobs from Jooble: {e}")
        return []


def fetch_adzuna_jobs(role, location):
    """Fetch job listings using Adzuna API."""
    try:
        url = f"https://api.adzuna.com/v1/api/jobs/in/search/1"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "results_per_page": 10,
            "what": role,
            "where": location
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        jobs = [
            {
                "title": job.get("title", "No Title"),
                "description": job.get("description", "No Description"),
                "link": job.get("redirect_url", ""),
                "source": "Adzuna"
            }
            for job in data.get("results", [])
        ]
        return jobs
    except requests.RequestException as e:
        print(f"Error fetching jobs from Adzuna: {e}")
        return []


def extract_required_skills(description):
    """Extract skills from job descriptions using NER model and spaCy."""
    try:
        doc = nlp(description.lower())
        extracted_skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]
        entities = skill_extractor(description)
        ner_skills = [entity['word'] for entity in entities if entity.get('entity_group') == "SKILL"]
        all_skills = list(set(extracted_skills + ner_skills))
        required_skills = [skill.lower() for skill in all_skills if skill.lower() in KNOWN_SKILLS]
        return required_skills
    except Exception as e:
        print(f"Error in skill extraction: {e}")
        return []


def find_skill_gaps(user_skills, job_skills):
    """Identify missing skills by comparing user skills with job requirements."""
    return list(set(job_skills) - set(user_skills.keys()))


def store_jobs_in_json(jobs, file_path):
    """Save job data to a JSON file."""
    with open(file_path, "w") as json_file:
        json.dump(jobs, json_file, indent=4)


def load_jobs_from_json(file_path):
    """Load job data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    return []


def get_user_skill_ratings():
    """Prompt the user to input skills and their ratings."""
    print("Enter your skills with ratings (format: skill:rating, e.g., python:4, java:3): ")
    skills_input = input().strip()
    skill_ratings = {}
    try:
        for item in skills_input.split(','):
            skill, rating = item.split(':')
            skill = skill.strip().lower()
            rating = int(rating.strip())
            if 1 <= rating <= 5:
                skill_ratings[skill] = rating
            else:
                print(f"Invalid rating for skill '{skill}'. Please enter a rating between 1 and 5.")
                return None
    except ValueError:
        print("Invalid input format. Use skill:rating, e.g., python:4, java:3.")
        return None
    return skill_ratings


# Main function
def main():
    try:
        location = input("Enter your preferred location: ").strip()
        preferred_role = input("Enter your preferred job role: ").strip()
        user_skill_ratings = get_user_skill_ratings()

        if not user_skill_ratings:
            print("Failed to parse skills and ratings. Exiting.")
            return

        jobs = load_jobs_from_json(TEMP_JOB_FILE)
        if not jobs:
            google_jobs = fetch_google_jobs([preferred_role, "software engineer"], location)
            jooble_jobs = fetch_jooble_jobs()
            adzuna_jobs = fetch_adzuna_jobs(preferred_role, location)
            jobs = google_jobs + jooble_jobs + adzuna_jobs
            store_jobs_in_json(jobs, TEMP_JOB_FILE)

        print(f"Total jobs loaded: {len(jobs)}")

        for job in jobs:
            job['required_skills'] = extract_required_skills(job.get("description", ""))

        all_job_skills = set(skill for job in jobs for skill in job.get("required_skills", []))
        skill_gaps = find_skill_gaps(user_skill_ratings, all_job_skills)

        skill_frequency = Counter(skill for job in jobs for skill in job.get("required_skills", []))
        ranked_skill_gaps = sorted(skill_gaps, key=lambda skill: skill_frequency[skill], reverse=True)

        print("\nSkill Gaps Identified:")
        print(ranked_skill_gaps)

        print("\nRecommended Jobs:")
        for job in jobs[:5]:
            print(f"Title: {job.get('title')}")
            print(f"Description: {job.get('description')[:200]}...")
            print(f"Link: {job.get('link')}")
            print(f"Source: {job.get('source')}")
            print("---")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
