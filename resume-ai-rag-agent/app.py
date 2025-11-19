import streamlit as st
from utils.parse_pdf import extract_text_from_pdf
from utils.clean_text import clean_text
from utils.chunk_text import chunk_text
from utils.embeddings import embed_chunks
from utils.ats_scoring import ats_score
from utils.rag_engine import build_faiss_index, query_faiss

import os
from openai import OpenAI


# ============================
# 🎨 Custom Professional CSS
# ============================
custom_css = """
<style>

    /* Global UI Styling */
    body {
        font-family: 'Inter', sans-serif;
    }

    .main {
        padding: 1.5rem;
    }

    /* Titles */
    h1, h2, h3 {
        font-weight: 700 !important;
    }

    /* Card Layout */
    .card {
        padding: 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.07);
        margin-bottom: 20px;
        border: 1px solid #e6e6e6;
    }

    /* Buttons */
    .stButton>button {
        background-color:#0066ff;
        color:white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color:#004fcc;
        color:white;
        border:none;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.2rem;
        border-radius: 10px 10px 0 0;
        background-color: #f0f2f6;
        font-weight: 600;
        color: #333;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0066ff !important;
        color: white !important;
    }

    textarea, input, select * {
        border-radius: 8px !important;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ============================
# 🔑 OpenAI Client
# ============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.openai.com/v1"
)


def answer_question_with_openai(query, retrieved_chunks):
    combined_context = "\n\n".join([c for c in retrieved_chunks])

    system_prompt = """
    You are an AI Resume Assistant.
    Use ONLY the provided resume chunks.
    If the answer is not available, say:
    "I couldn't find that information in your resume."
    """

    user_prompt = f"""
    Resume Context:
    {combined_context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================
# 📂 FILE UPLOAD SECTION
# ============================
st.set_page_config(page_title="Resume AI Assistant", layout="wide")
st.title("🧠 Resume AI Assistant (RAG + Multi-Agent System)")

with st.sidebar:
    st.markdown("## 📂 Upload Files")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_upload")
    uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd_upload")

    st.markdown("</div>", unsafe_allow_html=True)
    st.info("All tools unlock once you upload your resume.")

resume_clean = None
jd_clean = None

if uploaded_resume:
    resume_text = extract_text_from_pdf(uploaded_resume)
    resume_clean = clean_text(resume_text)

if uploaded_jd:
    jd_text = extract_text_from_pdf(uploaded_jd)
    jd_clean = clean_text(jd_text)


# ============================
# 📌 MAIN UI TABS
# ============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Resume Viewer",
    "🧠 Resume Q&A (RAG)",
    "✍️ Bullet Rewriters",
    "🎯 JD → Resume Tools",
    "💼 Career Tools",
    "📄 Export"
])


# ---------------- TAB 1 ----------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📄 Resume & JD Viewer")

    if resume_clean:
        st.subheader("Resume Text")
        st.write(resume_clean[:3000])

    if jd_clean:
        st.subheader("Job Description Text")
        st.write(jd_clean[:3000])

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 2 ----------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧠 Ask Questions About Your Resume")

    if resume_clean:
        question = st.text_input("Ask a question:", key="rag_question")

        if question:
            chunks = chunk_text(resume_clean)
            embeddings = embed_chunks(chunks)
            index = build_faiss_index(embeddings)
            matches = query_faiss(index, chunks, question)
            final_answer = answer_question_with_openai(question, matches)

            st.success(final_answer)

            with st.expander("Retrieved Resume Chunks"):
                for c in matches:
                    st.write(f"- {c}")
    else:
        st.warning("Upload your resume to enable this section.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 3 ----------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("✍️ Multi-Bullet Rewriter (STAR Method)")

    multi_bullet_input = st.text_area("Paste multiple bullet points:", height=200, key="multi_bullet_input")
    multi_target_role = st.text_input("Target Role:", key="multi_target_role")
    multi_style_option = st.selectbox(
        "Rewrite Style:",
        ["Professional", "Technical", "Impactful / FAANG", "Leadership-Focused"],
        key="multi_style"
    )

    if st.button("Rewrite All Bullets", key="rewrite_btn"):
        if multi_bullet_input.strip():
            system_prompt = """
            You are an expert FAANG resume rewriter using the STAR method.
            Rewrite EACH bullet clearly, concisely, and with measurable results.
            """
            user_prompt = f"""
            Bullets:
            {multi_bullet_input}

            Target Role: {multi_target_role}
            Style: {multi_style_option}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3
            )

            rewritten_output = response.choices[0].message.content
            st.success(rewritten_output)
        else:
            st.warning("Paste at least one bullet.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 4 ----------------
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🎯 JD → Resume Optimization Tools")

    if resume_clean and jd_clean:

        # 1. JD Tailoring Agent
        st.subheader("🎯 Tailored Resume Match")
        tailoring_question = st.text_input("Target Role:", key="tailor_role")

        if st.button("Analyze & Tailor Resume", key="tailor_btn"):
            prompt = f"""
            Compare resume vs JD and rewrite bullets using STAR.
            Resume: {resume_clean}
            JD: {jd_clean}
            Role: {tailoring_question}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "You tailor resumes for FAANG."},
                    {"role": "user", "content": prompt},
                ]
            )
            st.write(response.choices[0].message.content)

        st.markdown("---")

        # 2. Skill Gap Analyzer
        st.subheader("🧠 Skill Gap Analyzer")

        if st.button("Analyze Skill Gaps", key="gap_btn"):
            skill_prompt = f"""
            Compare resume vs JD and list:
            - Current skills  
            - Missing skills  
            - Recommendations  
            Resume: {resume_clean}
            JD: {jd_clean}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "ATS Skill Gap Analyzer"},
                    {"role": "user", "content": skill_prompt}
                ]
            )
            st.write(response.choices[0].message.content)

    else:
        st.warning("Upload both resume and JD to use this tab.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 5 ----------------
with tab5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("💼 Career Tools")

    # Resume Critique
    st.subheader("📊 Resume Critique")

    if resume_clean and st.button("Critique My Resume", key="critique_btn"):
        prompt = f"Critique resume professionally: {resume_clean}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Resume Reviewer"},
                {"role": "user", "content": prompt},
            ]
        )
        st.write(response.choices[0].message.content)

    st.markdown("---")

    # Cover Letter
    st.subheader("✉️ Cover Letter Generator")
    target_company = st.text_input("Company:", key="cover_company")
    target_position = st.text_input("Position:", key="cover_position")

    if resume_clean and jd_clean and st.button("Generate Cover Letter", key="cover_btn"):
        cover_prompt = f"""
        Write a cover letter for: {target_position} at {target_company}
        Based on resume: {resume_clean}
        JD: {jd_clean}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Cover Letter Writer"},
                {"role": "user", "content": cover_prompt},
            ]
        )
        st.write(response.choices[0].message.content)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 6 ----------------
with tab6:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📄 Export Your Results")

    export_text = st.text_area("Paste any content to export:", height=200, key="export_text")

    if st.button("Download as Text File", key="export_btn"):
        if export_text.strip():
            st.download_button(
                "Download File",
                export_text,
                "resume_output.txt",
                mime="text/plain"
            )
        else:
            st.warning("Paste something before downloading.")

    st.markdown("</div>", unsafe_allow_html=True)
