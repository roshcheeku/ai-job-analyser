import requests
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the English NLP model for skill extraction
nlp = spacy.load("en_core_web_sm")

# API Configuration
GOOGLE_API_KEY = "AIzaSyDd2Njm34U_vU_uYOPqFjI0S_yrUOMLhyo"  
CUSTOM_SEARCH_ENGINE_ID = "3741929af179a4b17"
JOOBLE_API_URL = "https://jooble.org/api/3f399c3d-2800-474c-bdba-4195187347eb"

# Step 1: Load Kaggle Dataset
def load_kaggle_dataset(file_path, description_column, category_column):
    print("\n--- Loading Kaggle Dataset ---")
    df = pd.read_csv(file_path)

    # Print column names to check
    print("Columns in dataset:", df.columns)
    
    job_descriptions = df[description_column].dropna().tolist()
    job_categories = df[category_column].dropna().tolist()

    print(f"Loaded {len(job_descriptions)} job descriptions and {len(job_categories)} categories.")
    return job_descriptions, job_categories

# Step 2: Fetch Job Postings from Google Custom Search API
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

            jobs = [({
                "title": item.get("title", "No Title"),
                "description": item.get("snippet", "No Description"),
                "link": item.get("link", ""),
                "source": "Google Search"
            }) for item in search_results.get("items", [])]

            all_jobs.extend(jobs)
        
        except requests.RequestException as e:
            print(f"Error fetching jobs for query '{query}': {e}")
    
    return all_jobs

# Step 3: Fetch Job Postings from Jooble API
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

# Step 4: Extract Skills from Job Description
def extract_required_skills(description):
    doc = nlp(description.lower())
    required_skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]
    return required_skills

# Step 5: Recommend Jobs Based on User Skills
def recommend_jobs(user_skills, job_data):
    recommended_jobs = []
    skills = [skill.strip().lower() for skill in user_skills.split(',')]
    
    for job in job_data:
        job_description = job.get("description", "").lower()
        job_title = job.get("title", "").lower()
        
        job_required_skills = extract_required_skills(job_description)
        
        skill_matches = [skill for skill in skills if skill in job_required_skills]
        skill_gaps = [skill for skill in job_required_skills if skill not in skills]
        
        if skill_matches:
            job['skill_match_count'] = len(skill_matches)
            job['skill_matches'] = skill_matches
            job['skill_gaps'] = skill_gaps
            recommended_jobs.append(job)
    
    recommended_jobs.sort(key=lambda x: x.get('skill_match_count', 0), reverse=True)
    
    return recommended_jobs

# Step 6: Train Model on Kaggle Dataset with Cross-Validation
def train_and_evaluate_model(job_descriptions, job_categories):
    print("\n--- Training and Evaluating Model ---")
    # Initialize model pipeline
    pipeline = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
    
    # Stratified K-Fold Cross Validation (preserves class balance in splits)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, job_descriptions, job_categories, cv=skf, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")

    # Optionally, train on the full training set and evaluate the final model
    pipeline.fit(job_descriptions, job_categories)
    y_pred = pipeline.predict(job_descriptions)
    
    print("\nClassification Report on Full Data:")
    print(classification_report(job_categories, y_pred))

# Step 7: Plot Validation Curve for TF-IDF max_features
def plot_validation_curve(job_descriptions, job_categories):
    print("\n--- Plotting Validation Curve ---")

    # Set the range of max_features to test
    param_range = [500, 1000, 2000, 5000, 10000]

    # Get validation scores
    train_scores, test_scores = validation_curve(
        TfidfVectorizer(), job_descriptions, job_categories, 
        param_name="max_features", param_range=param_range, 
        scoring="accuracy", cv=5
    )

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores.mean(axis=1), label="Training score", color="blue")
    plt.plot(param_range, test_scores.mean(axis=1), label="Cross-validation score", color="red")
    plt.title("Validation Curve for max_features in TF-IDF Vectorizer")
    plt.xlabel("max_features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Pipeline Execution
def main():
    # Load Kaggle dataset
    kaggle_file_path = "job_descriptions.csv"  # Adjust path
    description_column = "Job Description"  # Replace with actual column name
    category_column = "Role"  # Adjust to the correct column name (e.g., 'Role' or 'Category')
    job_descriptions, job_categories = load_kaggle_dataset(kaggle_file_path, description_column, category_column)

    # Train and evaluate model with cross-validation
    train_and_evaluate_model(job_descriptions, job_categories)

    # Plot validation curve for TF-IDF feature max_features
    plot_validation_curve(job_descriptions, job_categories)

    # Fetch real-time job data
    google_jobs = fetch_google_jobs()
    jooble_jobs = fetch_jooble_jobs()

    job_data = google_jobs + jooble_jobs
    print(f"Total jobs fetched: {len(job_data)}")

    # Get user skills
    user_skills = input("Enter your skills (comma-separated): ").strip()

    # Get recommended jobs
    recommended_jobs = recommend_jobs(user_skills, job_data)

    # Display top 10 recommended jobs
    print("\nRecommended Jobs:")
    for job in recommended_jobs[:10]:
        print(f"Title: {job.get('title')}")
        print(f"Description: {job.get('description')[:300]}...")
        print(f"Skill Matches: {job.get('skill_match_count', 0)}")
        print(f"Matched Skills: {', '.join(job.get('skill_matches', []))}")
        print(f"Skill Gaps: {', '.join(job.get('skill_gaps', []))}")
        print(f"Source: {job.get('source')}")
        print(f"Link: {job.get('link')}")
        print("---")

# Run the main function
if __name__ == "__main__":
    main()
