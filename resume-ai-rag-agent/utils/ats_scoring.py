import re
import numpy as np

def ats_score(resume_text, job_text):
    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    job_words = set(re.findall(r'\w+', job_text.lower()))
    matched = resume_words.intersection(job_words)
    return (len(matched) / len(job_words)) * 100
