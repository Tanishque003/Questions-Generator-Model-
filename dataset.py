import pandas as pd
import random

# Sample job titles and question templates
job_titles = ["Data Scientist", "Software Engineer", "Machine Learning Engineer", "Web Developer", "Project Manager", "Data Analyst"]
question_templates = [
    "What is your experience with {technology}?",
    "Explain a challenging project you worked on in {field}.",
    "How do you handle {situation} in your role as a {job_title}?",
]

# Generate unique questions
data = []

for job_title in job_titles:
    for i in range(1000000):  # Adjust this number as needed
        technology = random.choice(["Python", "Java", "SQL", "TensorFlow", "JavaScript"])
        field = random.choice(["data analysis", "web development", "machine learning", "software development"])
        situation = random.choice(["tight deadlines", "team conflicts", "technical challenges"])
        
        question = random.choice(question_templates).format(technology=technology, field=field, situation=situation, job_title=job_title)
        answer = "Expected answer for the question about {}".format(job_title)  # Placeholder for expected answers
        
        data.append({"job_title": job_title, "question": question, "answer": answer})

# Create a DataFrame and save as CSV
df = pd.DataFrame(data)

# Save as train, validation, and test datasets
df[:800000].to_csv("train2.csv", index=False)
df[800000:900000].to_csv("validation2.csv", index=False)
df[900000:].to_csv("test2.csv", index=False)
