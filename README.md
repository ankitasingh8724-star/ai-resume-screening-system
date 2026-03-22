# AI Resume Screening System

Python app that extracts text from resumes (PDF or TXT), normalizes it, surfaces skills via a keyword lexicon, and scores each resume against a job description using **TF-IDF** and **cosine similarity**. A **Streamlit** dashboard shows match scores, extracted skills, and a **ranked** candidate list.

## Project layout

```
resume-screening-ai/   # or your repo root
├── app.py             # Streamlit entrypoint
├── requirements.txt
├── data/
│   └── skills_keywords.txt   # editable skill phrases (optional; defaults exist)
├── models/            # reserved for saved models / vectorizers
└── src/
    ├── extract_text.py
    ├── preprocess.py
    ├── skills.py
    └── matcher.py
```

## Setup

```bash
cd "/path/to/AI resume screening system"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

1. Paste a **job description**.
2. Upload one or more **PDF** or **TXT** resumes.
3. Click **Add uploads to queue** (repeat if needed).
4. Click **Run screening** to see similarity scores, skills, and ranking.

## Notes

- PDF text uses **pdfminer.six** first, then **PyPDF2** as a fallback.
- Skill extraction is **lexicon-based** (NLP-style normalization + phrase matching). Extend `data/skills_keywords.txt` for your domain.
- Matching uses **scikit-learn** `TfidfVectorizer` (1–2 grams, English stop words) and **cosine similarity** between the job description and each resume in a shared vector space.

## Requirements

- Python 3.10+ recommended
