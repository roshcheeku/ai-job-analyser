import json
import os
from transformers import pipeline
import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")


# API Configuration  
GOOGLE_API_KEY = "AIzaSyDd2Njm34U_vU_uYOPqFjI0S_yrUOMLhyo"  
CUSTOM_SEARCH_ENGINE_ID = "3741929af179a4b17"  # Replace with your Custom Search Engine ID

# Initialize NER pipeline for skill extraction
skill_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Temporary JSON file for storing fetched jobs
TEMP_JOB_FILE = "fetched_jobs.json"

# Fetch jobs from Google Custom Search
def fetch_google_jobs(queries, location=""):
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

            jobs = [dict(title=item.get("title", "No Title"),
                         description=item.get("snippet", "No Description"),
                         link=item.get("link", ""),
                         source="Google Search")
                    for item in search_results.get("items", [])]
            all_jobs.extend(jobs)

        except requests.RequestException as e:
            print(f"Error fetching jobs for query '{query}': {e}")

    return all_jobs

# Extract required skills from job descriptions
def extract_required_skills(description):
    try:
        entities = skill_extractor(description)
        required_skills = [entity['word'] for entity in entities if entity.get('entity_group') == "SKILL"]
        return required_skills
    except Exception as e:
        print(f"Error in skill extraction: {e}")
        return []

# Identify skill gaps
def find_skill_gaps(user_skills, job_skills):
    skill_gaps = {}
    for job, skills in job_skills.items():
        gaps = list(set(skills) - set(user_skills.keys()))
        if gaps:
            skill_gaps[job] = gaps
    return skill_gaps

# Fetch certification courses for missing skills
def fetch_certification_courses(missing_skills):
    courses = {}
    for skill in missing_skills:
        try:
            params = {
                'key': GOOGLE_API_KEY,
                'cx': CUSTOM_SEARCH_ENGINE_ID,
                'q': f"{skill} certification course",
                'num': 5
            }
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            search_results = response.json()

            courses[skill] = [dict(title=item.get("title", "No Title"),
                                   link=item.get("link", ""))
                              for item in search_results.get("items", [])]

        except requests.RequestException as e:
            print(f"Error fetching courses for skill '{skill}': {e}")
            courses[skill] = []

    return courses

# Clustering job descriptions using Mini-Batch K-Means
def cluster_job_descriptions(jobs, n_clusters=5):
    descriptions = [job['description'] for job in jobs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    for i, job in enumerate(jobs):
        job['cluster'] = int(clusters[i])
    
    return jobs, kmeans

# Categorizing jobs using Naive Bayes
def categorize_jobs(jobs, categories):
    descriptions = [job['description'] for job in jobs]
    labels = np.random.choice(categories, len(descriptions))  # Random labels for demonstration

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    model = MultinomialNB()
    model.fit(tfidf_matrix, encoded_labels)

    predictions = model.predict(tfidf_matrix)
    for i, job in enumerate(jobs):
        job['category'] = label_encoder.inverse_transform([predictions[i]])[0]

    return jobs

# Skill ratings input
def get_user_skill_ratings():
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

# JSON helpers
def store_jobs_in_json(jobs, file_path):
    with open(file_path, "w") as json_file:
        json.dump(jobs, json_file, indent=4)

def load_jobs_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    return []

# Main function
def main():
    try:
        location = input("Enter your preferred location: ").strip()
        preferred_role = input("Enter your preferred job role: ").strip()
        user_skill_ratings = get_user_skill_ratings()

        if not user_skill_ratings:
            print("Failed to parse skills and ratings. Exiting.")
            return

        # Load or fetch jobs
        jobs = load_jobs_from_json(TEMP_JOB_FILE)
        if not jobs:
            queries = [preferred_role, "software engineer", "python developer", "remote jobs"]
            jobs = fetch_google_jobs(queries, location)
            store_jobs_in_json(jobs, TEMP_JOB_FILE)

        print(f"Total jobs loaded: {len(jobs)}")

        # Cluster jobs
        jobs, kmeans_model = cluster_job_descriptions(jobs)
        print(f"Jobs clustered into {kmeans_model.n_clusters} groups.")

        # Categorize jobs
        categories = ["Software", "Management", "Marketing", "Finance", "Healthcare"]
        jobs = categorize_jobs(jobs, categories)
        print("Job categorization completed.")

        # Extract required skills
        job_skills = {}
        for job in jobs:
            required_skills = extract_required_skills(job.get("description", ""))
            job_skills[job['title']] = required_skills

        # Skill gaps
        skill_gaps = find_skill_gaps(user_skill_ratings, job_skills)
        print("\nSkill Gaps Identified:")
        for job, gaps in skill_gaps.items():
            print(f"Job: {job}")
            print(f"Required Skills: {', '.join(job_skills[job])}")
            print(f"Missing Skills: {', '.join(gaps)}")
            print("---")

        # Certification recommendations
        missing_skills = set(skill for gaps in skill_gaps.values() for skill in gaps)
        certification_courses = fetch_certification_courses(missing_skills)
        print("\nRecommended Certification Courses:")
        for skill, courses in certification_courses.items():
            print(f"\nSkill: {skill}")
            for course in courses:
                print(f"- {course['title']} ({course['link']})")

        # Display jobs
        print("\nRecommended Jobs:")
        for job in jobs[:10]:  # Show top 10 jobs
            print(f"Title: {job.get('title')}")
            print(f"Description: {job.get('description')[:200]}...")
            print(f"Category: {job.get('category')}")
            print(f"Cluster: {job.get('cluster')}")
            print(f"Link: {job.get('link')}")
            print("---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
