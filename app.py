import re
import numpy as np
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import os


# ─── File parsers ─────────────────────────────────────────────────────────────
def read_pdf(file_bytes):
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception:
        return ""

def read_docx(file_bytes):
    try:
        import docx
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception:
        return ""

def clean_text(t):
    t = (t or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", t).strip()

# ─── Model + data loading ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_job_data():
    try:
        jobs_emb = np.load("jobs_emb.npy")
        jobs_meta = pd.read_csv("jobs_metadata.csv").fillna("")
        return jobs_emb, jobs_meta
    except FileNotFoundError:
        return None, None

# ─── Keyword analysis ─────────────────────────────────────────────────────────
def top_keywords(job_text, resume_text, top_k=30):
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform([job_text, resume_text])
        terms = np.array(vec.get_feature_names_out())
        job_scores = X[0].toarray().ravel()
        top_idx = np.argsort(job_scores)[::-1][:top_k]
        kw = terms[top_idx]
        resume_lower = resume_text.lower()
        present = [k for k in kw if k.lower() in resume_lower]
        missing = [k for k in kw if k.lower() not in resume_lower]
        return present, missing
    except Exception:
        return [], []

# ─── Core scoring ─────────────────────────────────────────────────────────────
def score_resume_vs_job(resume_text, job_text, model):
    def chunk_text(text, size=150, overlap=30):
        words = text.split()
        if len(words) <= size:
            return [text]
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + size]))
            i += size - overlap
        return chunks

    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_text)

    all_texts = resume_chunks + job_chunks
    all_embs = model.encode(
        all_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    r_embs = all_embs[:len(resume_chunks)]
    j_embs = all_embs[len(resume_chunks):]

    sim_matrix = cosine_similarity(j_embs, r_embs)
    best_per_job = sim_matrix.max(axis=1)
    embedding_score = float(best_per_job.mean())

    present, missing = top_keywords(job_text, resume_text, top_k=30)
    total = len(present) + len(missing)
    keyword_score = len(present) / total if total > 0 else 0.0

    final_score = (0.65 * embedding_score) + (0.35 * keyword_score)
    return final_score, present, missing

# ─── Resume tailoring advice ──────────────────────────────────────────────────
def generate_tailoring_advice(present, missing, resume_text, job_text):
    """
    Generate specific, actionable advice on how to tailor the resume
    to better match the job description.
    """
    advice = []

    # Section 1: Keywords to weave in
    if missing:
        advice.append({
            "title": "🔑 Keywords to Add to Your Resume",
            "content": f"The following terms appear in the job description but not in your resume. Work these into your bullet points, skills section, or summary naturally:\n\n**{', '.join(missing[:15])}**",
            "type": "keywords"
        })

    # Section 2: Specific bullet point rewrites
    if missing:
        rewrites = []
        for kw in missing[:5]:
            rewrites.append(f"- Instead of a generic bullet, try: *'Leveraged **{kw}** to [specific outcome], resulting in [metric/impact]'*")
        advice.append({
            "title": "✏️ How to Rewrite Your Bullet Points",
            "content": "Tailor your existing experience bullets to mirror the job's language:\n\n" + "\n".join(rewrites),
            "type": "bullets"
        })

    # Section 3: Skills section advice
    missing_skills = [k for k in missing if len(k.split()) == 1][:8]
    if missing_skills:
        advice.append({
            "title": "🛠️ Update Your Skills Section",
            "content": f"Add these specific skills to your skills section if you have experience with them:\n\n**{', '.join(missing_skills)}**\n\nEven if you have basic exposure, listing them helps pass ATS (Applicant Tracking Systems) filters.",
            "type": "skills"
        })

    # Section 4: Summary/objective rewrite
    if missing:
        top3 = missing[:3]
        advice.append({
            "title": "📝 Rewrite Your Summary/Objective",
            "content": f"Your resume summary should mirror the job's priorities. Try opening with something like:\n\n*'Results-driven professional with experience in **{top3[0]}**{f', **{top3[1]}**' if len(top3) > 1 else ''}{f', and **{top3[2]}**' if len(top3) > 2 else ''}, seeking to [role goal from job description]...'*",
            "type": "summary"
        })

    # Section 5: What's already working
    if present:
        advice.append({
            "title": "✅ What's Already Working",
            "content": f"These keywords from the job description are already present in your resume — keep them prominent:\n\n**{', '.join(present[:12])}**",
            "type": "positive"
        })

    # Section 6: ATS tip
    advice.append({
        "title": "🤖 ATS Tip",
        "content": "Most companies use Applicant Tracking Systems that scan for exact keyword matches before a human ever reads your resume. Make sure the job title from the posting appears somewhere in your resume, and avoid putting key information only in headers, footers, or tables — ATS systems often can't read those.",
        "type": "ats"
    })

    return advice

def interpret(score_pct):
    if score_pct >= 60:
        return ("Excellent Match", "🌟", "#a6e3a1", "#1a2e1a")
    elif score_pct >= 50:
        return ("Good Match", "✅", "#89b4fa", "#1a1e2e")
    elif score_pct >= 38:
        return ("Partial Match", "🟡", "#f9e2af", "#2e2a1a")
    else:
        return ("Poor Match", "🔴", "#f38ba8", "#2e1a1a")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Reviewer", layout="wide")

st.markdown("""
<style>
    .verdict-card {
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    .verdict-emoji { font-size: 56px; margin-bottom: 12px; }
    .verdict-label { font-size: 36px; font-weight: 800; margin-bottom: 8px; }
    .verdict-sub { font-size: 16px; opacity: 0.85; }
    .advice-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .advice-title {
        font-size: 17px;
        font-weight: 700;
        color: #cdd6f4;
        margin-bottom: 10px;
    }
    .advice-body {
        font-size: 14px;
        color: #a6adc8;
        line-height: 1.7;
    }
    .tag-present {
        display: inline-block;
        background: #a6e3a122;
        color: #a6e3a1;
        border: 1px solid #a6e3a144;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 13px;
    }
    .tag-missing {
        display: inline-block;
        background: #f38ba822;
        color: #f38ba8;
        border: 1px solid #f38ba844;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 13px;
    }
    .dataset-badge {
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Match")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Top N job matches (dataset mode)", 1, 10, 5)
    show_debug = st.checkbox("Show debug preview", value=False)

# ─── Load resources ───────────────────────────────────────────────────────────
model = load_model()
jobs_emb, jobs_meta = load_job_data()

if jobs_emb is None:
    st.warning("⚠️ `jobs_emb.npy` and `jobs_metadata.csv` not found. Run `backend.py` first, then relaunch.")

# ─── Input section ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resume")
    resume_mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)
    resume_text = ""
    if resume_mode == "Paste text":
        resume_text = st.text_area("Paste your resume", height=320, placeholder="Paste resume content here...")
    else:
        up = st.file_uploader("Upload (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
        if up:
            raw = up.read()
            if up.name.lower().endswith(".pdf"):
                resume_text = read_pdf(raw)
            elif up.name.lower().endswith(".docx"):
                resume_text = read_docx(raw)
            else:
                resume_text = raw.decode("utf-8", errors="ignore")
            if resume_text:
                st.success(f"Loaded {len(resume_text.split())} words from {up.name}")
            else:
                st.error("Could not extract text from file. Try copy-pasting instead.")

with col2:
    st.subheader("Job Description (optional)")
    st.caption("Paste a specific job to compare directly — or leave blank to search your full dataset.")
    job_text = st.text_area("Paste job description", height=320, placeholder="Paste job posting here...")

resume_text = clean_text(resume_text)
job_text = clean_text(job_text)

run = st.button("Analyze", type="primary", use_container_width=True)

# ─── Analysis ─────────────────────────────────────────────────────────────────
if run:
    if not resume_text:
        st.error("Please provide your resume text.")
        st.stop()

    if len(resume_text.split()) < 20:
        st.error("Resume text is too short — make sure the file was parsed correctly.")
        st.stop()

    st.markdown("---")

    # ── Mode A: specific job description pasted ───────────────────────────────
    if job_text:
        if len(job_text.split()) < 10:
            st.error("Job description is too short to analyze.")
            st.stop()

        with st.spinner("Analyzing match..."):
            final_score, present, missing = score_resume_vs_job(resume_text, job_text, model)

        score_pct = round(final_score * 100, 1)
        label, emoji, text_color, bg_color = interpret(score_pct)

        # Verdict card
        st.markdown(f"""
        <div class="verdict-card" style="background:{bg_color}; border: 2px solid {text_color}55;">
            <div class="verdict-emoji">{emoji}</div>
            <div class="verdict-label" style="color:{text_color};">{label}</div>
            <div class="verdict-sub" style="color:{text_color};">
                This resume is a <strong>{label.lower()}</strong> for this job description.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Keyword tags ──────────────────────────────────────────────────────
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(f"✅ Keywords Present ({len(present)})")
            if present:
                tags = "".join(f'<span class="tag-present">{k}</span>' for k in present[:25])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.write("None detected.")
        with right:
            st.subheader(f"❌ Keywords Missing ({len(missing)})")
            if missing:
                tags = "".join(f'<span class="tag-missing">{k}</span>' for k in missing[:25])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.write("Great — most key terms found!")

        st.markdown("---")

        # ── Tailoring advice ──────────────────────────────────────────────────
        st.subheader("📋 How to Tailor Your Resume for This Job")
        st.caption("Specific, actionable steps to improve your match for this role.")

        advice_list = generate_tailoring_advice(present, missing, resume_text, job_text)

        for advice in advice_list:
            with st.expander(advice["title"], expanded=True):
                st.markdown(advice["content"])

    # ── Mode B: search the full pre-computed dataset ──────────────────────────
    elif jobs_emb is not None:
        with st.spinner(f"Searching {len(jobs_emb):,} jobs in your dataset..."):
            resume_emb_vec = model.encode(
                [resume_text],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]
            sims = cosine_similarity(resume_emb_vec.reshape(1, -1), jobs_emb)[0]
            top_idx = np.argsort(sims)[::-1][:top_n]

        st.subheader(f"Top {top_n} Matching Jobs from Your Dataset")
        st.caption("Paste a job description on the right to get tailoring advice for a specific role.")

        for rank, idx in enumerate(top_idx, 1):
            raw_sim = sims[idx]
            row = jobs_meta.iloc[idx]
            job_master = row.get("master_text", "")

            present, missing = top_keywords(job_master, resume_text, top_k=30)
            total = len(present) + len(missing)
            kw_score = len(present) / total if total > 0 else 0.0
            blended = round(((0.65 * raw_sim) + (0.35 * kw_score)) * 100, 1)

            label, emoji, text_color, bg_color = interpret(blended)

            with st.expander(f"#{rank} — {row.get('title', 'N/A')}  |  {emoji} {label}"):
                st.markdown(f"""
                <div class="dataset-badge" style="background:{bg_color}; color:{text_color}; border: 1px solid {text_color}55;">
                    {emoji} {label}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Skills:** {row.get('skills', '')[:300]}")
                st.markdown(f"**Description:** {row.get('description', '')[:400]}...")

                # Tailoring advice for this job
                st.markdown("---")
                st.markdown("**How to tailor your resume for this role:**")
                advice_list = generate_tailoring_advice(present, missing, resume_text, job_master)
                for advice in advice_list[:3]:  # Show top 3 pieces of advice in dataset mode
                    st.markdown(f"**{advice['title']}**")
                    st.markdown(advice["content"])
                    st.markdown("")

    else:
        st.error("No job description pasted and no dataset loaded. Run `backend.py` first.")

    # ── Debug ─────────────────────────────────────────────────────────────────
    if show_debug:
        st.markdown("---")
        st.subheader("Debug")
        st.caption(f"Resume: {len(resume_text.split())} words")
        st.write(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
        if job_text:
            st.caption(f"Job: {len(job_text.split())} words")
            st.write(job_text[:1000] + ("..." if len(job_text) > 1000 else ""))

else:
    st.info("Upload your resume and optionally paste a job description, then click **Analyze**.")