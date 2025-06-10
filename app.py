import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ---------- Load environment variables ----------
load_dotenv()
api_key = "AIzaSyBMjbeinGwyAcWpLNgOqIOw-BGlJKZ-zMo"
genai.configure(api_key=api_key)

# ---------- Page configuration ----------
st.set_page_config(page_title="EduGenie Bot", layout="wide")

# ---------- Load university data ----------
@st.cache_data
def load_university_data():
    df = pd.read_csv("cwurData.csv")
    df["description"] = df["institution"] + ", " + df["country"] + ", Rank: " + df["world_rank"].astype(str)
    return df

df = load_university_data()

# ---------- Load sentence transformer model and FAISS index ----------
@st.cache_resource
def setup_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = df["description"].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings).astype("float32"))
    return model, index

model, index = setup_model_and_index()

# ---------- University search ----------
def search_universities(query, top_n=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_n * 2)  # Overfetch for deduplication
    results = df.iloc[I[0]].drop_duplicates(subset=["institution"]).head(top_n)
    return results

# ---------- Gemini response ----------
def ask_gemini(query, user_profile, matches_df):
    prompt = f"""
You are an intelligent academic consultant named EduGenie.

A student is looking for university options. Analyze their profile and recommend the best choices based on:
- Country preference
- Area of interest
- CGPA
- World Rank

Profile:
Degree: {user_profile['degree']}
CGPA: {user_profile['cgpa']}
Interest: {user_profile['interest']}
Preferred Countries: {', '.join(user_profile['preferred_countries'])}

Top matched universities:
{matches_df[['institution', 'country', 'world_rank']].to_string(index=False)}

Now, return a full academic consultation with the following structure:
1. 📍 Top University Recommendations (3 universities)
2. 💼 Career Paths (5 roles based on their interest)
3. 🗺️ Roadmap to Become a Machine Learning Engineer (if interest is AI)
"""
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text

# ---------- Streamlit UI ----------
st.title("🎓 EduGenie: AI-Powered University Bot")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        degree = st.text_input("🎓 Degree", value="B.Tech")
        cgpa = st.number_input("📊 CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
    with col2:
        interest = st.text_input("🧠 Area of Interest", value="Artificial Intelligence")
        preferred_countries = st.multiselect("🌍 Preferred Countries", ["USA", "Canada", "Germany", "UK", "Australia"], default=["Germany", "Canada"])

    query = st.text_input("🔍 Search Query", value="Top universities for AI in Germany and Canada")
    submitted = st.form_submit_button("🚀 Get Recommendations")

if submitted:
    with st.spinner("🔎 Searching universities..."):
        matches = search_universities(query)
        st.subheader("🏫 Top University Matches")
        for _, row in matches.iterrows():
            st.markdown(f"- **{row['institution']}** — {row['country']} (World Rank: {row['world_rank']})")

    user_profile = {
        'degree': degree,
        'cgpa': cgpa,
        'interest': interest,
        'preferred_countries': preferred_countries
    }

    with st.spinner("🤖 Generating expert recommendations using Gemini..."):
        try:
            gemini_output = ask_gemini(query, user_profile, matches)
            st.markdown("---")
            st.markdown(gemini_output)
        except Exception as e:
            st.error(f"❌ Gemini API Error: {str(e)}")
