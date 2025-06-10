import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --------- Setup API key from .env or fallback ---------
load_dotenv()
api_key = "AIzaSyBMjbeinGwyAcWpLNgOqIOw-BGlJKZ-zMo"
genai.configure(api_key=api_key)

# --------- Load dataset ---------
@st.cache_data
def load_data():
    df = pd.read_csv("cwurData.csv")
    df["description"] = df["institution"] + ", " + df["country"] + ", Rank: " + df["world_rank"].astype(str)
    return df

df = load_data()

# --------- Setup model + FAISS ---------
@st.cache_resource
def setup_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode(df["description"].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeds[0].shape[0])
    index.add(np.array(embeds).astype("float32"))
    return model, index

embedder, index = setup_embeddings()

# --------- FAISS search ---------
def search_unis(query, top_n=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_n * 2)
    return df.iloc[I[0]].drop_duplicates(subset=["institution"]).head(top_n)

# --------- Gemini generation ---------
def ask_gemini(prompt, model_name):
    model = genai.GenerativeModel(model_name)
    return model.generate_content(prompt).text

# --------- Streamlit UI ---------
st.set_page_config(page_title="EduGenie AI Bot", layout="wide")
st.title("🎓 EduGenie: AI-Powered University Recommendation Assistant")

with st.sidebar:
    st.header("🛠️ Settings")
    selected_model = st.selectbox("Gemini Model", [
        "models/gemini-1.5-flash", 
        "models/gemini-1.5-pro-latest"
    ])
    st.caption("💡 Flash is faster & quota-friendly. Use Pro for more detailed responses.")

with st.form("profile_form"):
    st.subheader("👤 Your Academic Profile")
    col1, col2 = st.columns(2)
    with col1:
        degree = st.text_input("🎓 Degree", value="B.Tech")
        cgpa = st.number_input("📊 CGPA", 0.0, 10.0, 8.5, 0.1)
    with col2:
        interest = st.text_input("🧠 Area of Interest", value="Artificial Intelligence")
        countries = st.multiselect("🌍 Preferred Countries", ["USA", "Germany", "Canada", "UK", "Australia"], default=["Germany", "Canada"])
    
    query = st.text_input("🔍 Search Query", "Top AI universities in Germany and Canada")
    submitted = st.form_submit_button("🔎 Find Universities")

if submitted:
    with st.spinner("Searching universities..."):
        matched = search_unis(query)
    
    st.success("✅ Top University Matches Found!")
    with st.expander("🏫 Top University Matches"):
        for _, row in matched.iterrows():
            st.markdown(f"- **{row['institution']}** — {row['country']} (Rank: {row['world_rank']})")

    user_profile = {
        "degree": degree,
        "cgpa": cgpa,
        "interest": interest,
        "preferred_countries": countries
    }

    top_matches = matched.head(3)
    gemini_prompt = f"""
You are EduGenie, an AI academic advisor. Analyze this profile and suggest:

1. Top 3 university recommendations
2. Career paths
3. A roadmap to become a machine learning engineer

Profile:
Degree: {degree}
CGPA: {cgpa}
Interest: {interest}
Countries: {', '.join(countries)}

Top Matches:
{top_matches[['institution', 'country', 'world_rank']].to_string(index=False)}
"""

    if st.button("🚀 Generate AI Recommendations"):
        with st.spinner("Contacting Gemini..."):
            try:
                response = ask_gemini(gemini_prompt, selected_model)
                st.success("✅ Gemini response generated!")

                with st.expander("📍 University Recommendations"):
                    st.markdown(response.split("💼")[0])

                if "💼" in response:
                    with st.expander("💼 Career Paths"):
                        career_section = response.split("💼")[1].split("🗺️")[0]
                        st.markdown("💼 " + career_section)

                if "🗺️" in response:
                    with st.expander("🗺️ Roadmap to ML Engineer"):
                        roadmap = response.split("🗺️")[1]
                        st.markdown("🗺️ " + roadmap)

            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")
