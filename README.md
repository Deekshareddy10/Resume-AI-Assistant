# Resume-AI-Assistant
An advanced AI Resume Intelligence Platform built using RAG, FAISS, GPT-4o-mini &amp; Streamlit

Live Demo:
👉 https://huggingface.co/spaces/Deekshareddy10/resume-ai-rag-agent

### Overview

Resume AI Assistant is a full-stack, production-ready AI system that enhances résumés using cutting-edge LLM and RAG techniques. It provides:

✔ Resume Question-Answering (RAG)
✔ Multi-bullet rewriting using STAR
✔ Job Description → Resume tailoring
✔ Skill gap analysis & learning roadmap
✔ ATS match scoring
✔ Cover letter generation
✔ PDF parsing + semantic chunking + FAISS retrieval
✔ Professional, interactive UI

Built for FAANG-level preparation and real-world job applications.

## 🎯 Key Features
## 🔍 1. RAG-Based Resume Q&A

Ask any question about your résumé.
The system uses:

PDF → text extraction

Text cleaning

Chunking & embedding

FAISS vector search

GPT-4o-mini reasoning

Strictly non-hallucinatory answers

## ✨ 2. Multi-Bullet Resume Rewriter (STAR Method)

Paste multiple bullets → get high-impact, measurable, FAANG-style rewrites.

Supports tones:

Professional

Technical

Impactful

Leadership

## 🎯 3. Job Description → Resume Tailoring Agent

Automatically matches your résumé to any JD:

Extracts JD key responsibilities

Matches your relevant experience

Identifies missing skills

Rewrites bullets to align with JD

Follows STAR + ATS best practices

## 🧠 4. Skill Gap Analyzer

Compares résumé & JD and identifies:

Existing skills

Missing skills

Weak areas

Recommended tools/courses

Suggested projects

## ✉️ 5. Cover Letter Generator

Generates a personalized, structured, 3–4 paragraph FAANG-style cover letter based on:

Resume

Job description

Target company

Role

## 📊 6. ATS Match Scoring

Computes ATS compatibility using semantic & keyword overlap.

## 📄 7. Export Output

Download rewritten bullets, cover letters, or full tailored résumé text.

## 🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Deekshareddy10/Resume-AI-Assistant.git
cd Resume-AI-Assistant

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Your API Key

Create a .env file:

OPENAI_API_KEY=your_key_here

4️⃣ Run Locally
streamlit run app.py

🌐 Deployment

The application is deployed on Hugging Face Spaces:
👉 https://huggingface.co/spaces/Deekshareddy10/resume-ai-rag-agent

No server cost

## 🛠 Tech Stack

Python

Streamlit

FAISS

OpenAI GPT-4o-mini

Sentence Transformers

PyPDF2

dotenv

Hugging Face Spaces

## AI/ML Concepts Used:

Retrieval-Augmented Generation (RAG)

Semantic embeddings

Vector search (FAISS)

Multi-agent prompting

STAR method bullet rewriting

ATS optimization

## 🥇 Why This Project Stands Out

This project demonstrates end-to-end real-world LLM engineering:

Advanced RAG pipeline

Multi-agent LLM workflows

Full UI/UX

Real deployment

PDF processing + embeddings

FAISS vector search

ATS scoring logic

Tailoring & rewriting agents

Professional-level output

👩‍💻 Author

Deeksha Reddy Patlolla
Master’s in Computer Science @ University of Colorado Denver

