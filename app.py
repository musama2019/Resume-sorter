import zipfile
import os
import re
import pandas as pd
import PyPDF2
from flask import Flask, request, render_template, redirect, url_for, Response, stream_with_context
from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESUME_FOLDER'] = 'extracted_resumes'
top_candidates = []  # Global variable to hold the sorted candidates


def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_contact_info(resume_text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_matches = re.findall(email_pattern, resume_text)
    email = email_matches[0] if email_matches else 'Not Found'
    return email


def extract_resumes(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return [os.path.join(extract_path, file) for file in os.listdir(extract_path) 
            if file.endswith(".pdf") or file.endswith(".docx")]


def read_resume(file_path):
    resume_text = extract_text_from_pdf(file_path)
    email = extract_contact_info(resume_text)
    return resume_text, email


def read_job_description(file_path):
    return extract_text_from_pdf(file_path)


def load_rubrics(file_path):
    rubric_df = pd.read_excel(file_path)
    rubrics = {}

    for _, row in rubric_df.iterrows():
        criteria = row['Criteria']
        points_range = row['Points']
        min_points, max_points = map(int, points_range.split('-'))
        notes = row['Notes']
        weightage = row['Weightage']
        rubrics[criteria] = {'min_points': min_points, 'max_points': max_points, 'notes': notes, 'weightage': weightage}

    return rubrics


def score_resume_with_hf(resume_text, job_description, rubrics, model_pipeline):
    scoring_details = "\n".join([f"{criterion}: {details['notes']} (Score range: {details['min_points']}-{details['max_points']})" 
                                  for criterion, details in rubrics.items()])
    
    prompt = f"""
    You are an HR professional. Score the following resume based on the provided job description and rubrics.
    BUT KEEP IN MIND THAT SCORING SHOULD ONLY BE BASED ON RELEVANCE TO JOB DESCRIPTION.
    Keep the scoring strict on the basis of rubrics provided but relevance to job role is must.
    For each criterion, use the given notes and score within the range provided.
    \nJob Description: {job_description}
    \nRubric Notes: {scoring_details}
    \nResume: {resume_text}
    Please provide a score for each criterion within its respective range.
    Your output should be formatted like this:
    resume_scores = 
    'criterion 1': score assigned,
    justification: justification for criterion 1,
    'criterion 2': score assigned,
    justification: justification for criterion 2,
    ...
    Also note that in response, keep the names of criterions exactly similar to those in rubrics. Do not change their case sensitivity and also don't add any marks, underscores or any extra spaces from your own side. KEEP THEM AS IT IS.
    """
    
    response = model_pipeline.predict(prompt)  # Update to invoke
    score_text = response.strip()  
    scores = {}
    individual_scores = {}
    try:
        for criterion, details in rubrics.items():
            if criterion == "Total Possible Points":
                continue

            score_pattern = rf"'{re.escape(criterion)}'\s*:\s*([\d.]+)\s*,?"
            justification_pattern = rf"{re.escape(criterion)}\s*:\s*[\d.]+,\s*[Jj]ustification\s*:\s*(.*?)(?=\s*'|$)"

            match = re.search(score_pattern, score_text)
            if match:
                score = float(match.group(1))
                score = max(details['min_points'], min(score, details['max_points']))
                individual_scores[criterion] = score
                weighted_score = (score / details['max_points']) * details['weightage'] 
                scores[criterion] = weighted_score
            else:
                scores[criterion] = details['min_points'] * details['weightage']

    except Exception as e:
        print(f"Error parsing response for scores: {e}")

    total_score = sum(scores.values())
    
    return individual_scores, total_score


def process_resumes(resume_files, job_description, rubrics, model_pipeline):
    global top_candidates
    candidate_scores = []
    total_resumes = len(resume_files)

    for index, resume_file in enumerate(resume_files):
        resume_text, email = read_resume(resume_file)
        candidate_name = os.path.basename(resume_file)
        print(f"\nProcessing resume: {candidate_name}")
        individual_scores, total_score = score_resume_with_hf(resume_text, job_description, rubrics, model_pipeline)
        candidate_scores.append((candidate_name, total_score, individual_scores))

        yield f"Processed resume {index + 1}/{total_resumes}: {candidate_name}\n"
        print(f"Processed resume {index + 1}/{total_resumes}: {candidate_name}")

    top_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
    for file_path in resume_files:
        os.remove(file_path)  # Delete the folder if empty

        yield "COMPLETE\n"  # Signal for client to redirect



@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'resume_files' not in request.files or 'job_description' not in request.files or 'rubrics' not in request.files:
        return redirect(url_for('index'))
    
    resume_zip = request.files['resume_files']
    job_description_file = request.files['job_description']
    rubrics_file = request.files['rubrics']
    
    resume_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_zip.filename)
    job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], job_description_file.filename)
    rubrics_path = os.path.join(app.config['UPLOAD_FOLDER'], rubrics_file.filename)
    
    resume_zip.save(resume_zip_path)
    job_description_file.save(job_description_path)
    rubrics_file.save(rubrics_path)

    if not os.path.exists(app.config['RESUME_FOLDER']):
        os.makedirs(app.config['RESUME_FOLDER'])
    
    resume_files = extract_resumes(resume_zip_path, app.config['RESUME_FOLDER'])
    job_description = read_job_description(job_description_path)
    rubrics = load_rubrics(rubrics_path)

    model_pipeline = ChatGroq(temperature=0, model="gemma-7b-it", groq_api_key="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")

    return Response(stream_with_context(process_resumes(resume_files, job_description, rubrics, model_pipeline)), mimetype='text/plain')


@app.route('/results')
def results():
    return render_template('results.html', top_candidates=top_candidates)


if __name__ == '__main__':
    app.run(debug=True)
