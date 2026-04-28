"""
app.py — Streamlit Dashboard
─────────────────────────────────────────────────────────────────────────────
AI-Powered IT Career Intelligence & Resume Optimization System
Run: streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from api_client import APIClient
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Prevents Streamlit threading issues with pyplot
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.nlp_processor import NLPProcessor
from src.domain_matcher import DomainMatcher, DOMAIN_PROFILES
from src.skill_analyzer import SkillAnalyzer
from src.predictor import ResumePredictor
from src.explainer import Explainer

# ── API Client ────────────────────────────────────────────────────────────────
# Bug 8 Fix: Imported from api_client.py — single source of truth for all API calls.


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Career Intelligence",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS (Adaptive for Light/Dark Mode) ────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: gray; font-size: 1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        border-left: 4px solid #00c6ff;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-card-title { color: gray; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .metric-card-val { color: var(--text-color); font-size: 1.6rem; font-weight: 800; }
    .score-high  { color: #2ecc71; font-size: 2rem; font-weight: 800; }
    .score-mid   { color: #f39c12; font-size: 2rem; font-weight: 800; }
    .score-low   { color: #e74c3c; font-size: 2rem; font-weight: 800; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #00c6ff;
        border-bottom: 2px solid #00c6ff; padding-bottom: 0.3rem; margin: 1.5rem 0 1rem 0;
    }
    .skill-tag-green {
        display: inline-block; background: rgba(46, 204, 113, 0.15); color: #2ecc71;
        border: 1px solid #2ecc71; border-radius: 20px;
        padding: 2px 12px; margin: 3px; font-size: 0.82rem; font-weight: 600;
    }
    .skill-tag-red {
        display: inline-block; background: rgba(231, 76, 60, 0.15); color: #e74c3c;
        border: 1px solid #e74c3c; border-radius: 20px;
        padding: 2px 12px; margin: 3px; font-size: 0.82rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── session state "memory" initialization ─────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_email" not in st.session_state: st.session_state.user_email = ""
if "memory_skills" not in st.session_state: st.session_state.memory_skills = ""
if "memory_raw_text" not in st.session_state: st.session_state.memory_raw_text = ""
if "memory_exp" not in st.session_state: st.session_state.memory_exp = 1.0
if "memory_proj" not in st.session_state: st.session_state.memory_proj = 2
if "memory_edu" not in st.session_state: st.session_state.memory_edu = "Bachelor's"
if "memory_certs" not in st.session_state: st.session_state.memory_certs = 0
if "manual_extracted_skills" not in st.session_state: st.session_state.manual_extracted_skills = []
if "manual_skills_extracted" not in st.session_state: st.session_state.manual_skills_extracted = False
# Bug 4 Fix: Store JWT token in session so APIClient can attach it to requests
if "auth_token" not in st.session_state: st.session_state.auth_token = ""
# Stores the last prediction result so the Save button can live outside the Predict block
if "last_prediction_payload" not in st.session_state: st.session_state.last_prediction_payload = None
if "last_save_status" not in st.session_state: st.session_state.last_save_status = None


# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    processor = NLPProcessor()
    matcher = DomainMatcher()
    analyzer = SkillAnalyzer()
    predictor = ResumePredictor()

    # Bug 2 Fix: use __file__-relative paths so app works from any working directory
    BASE_DIR = Path(__file__).parent
    rf_path = BASE_DIR / "saved_models" / "rf_classifier.pkl"
    if rf_path.exists():
        predictor.load_model(str(rf_path), str(BASE_DIR / "saved_models" / "scaler.pkl"))

    vec_path = BASE_DIR / "saved_models" / "vectorizer.pkl"
    if vec_path.exists():
        processor.load_vectorizer(str(vec_path))

    return processor, matcher, analyzer, predictor

processor, matcher, analyzer, predictor = load_components()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/source-code.png", width=70)
    st.title("IT Career Intel")
    
    if st.session_state.logged_in:
        st.success(f"👤 {st.session_state.user_email}")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.auth_token = ""  # Bug 4 Fix: clear token on logout
            st.rerun()
        st.divider()
        page = st.radio(
            "📌 Navigation",
            ["🏠 Home", "🔍 Tech Skill Analysis", "📄 IT Resume Scorer", "📊 Tech Domain Explorer", "👤 My Profile"],
            index=0,
        )
    else:
        st.divider()
        page = st.radio("📌 Navigation", ["🔐 Login / Sign Up"])
        st.info("Please log in to access the career tools.")
    
    if st.session_state.logged_in and st.button("🗑️ Clear Session Data"):
        st.session_state.memory_skills = ""
        st.session_state.memory_raw_text = ""
        st.rerun()
        
    st.divider()
    st.caption("🛠 Built for IT Roles · ML Powered")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔐 Login / Sign Up":
    st.markdown('<p class="main-title">Welcome to IT Career Intel</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Log in to analyze your tech stack and save your ATS scores</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with tab1:
        st.subheader("Login to your account")
        log_email = st.text_input("Email", key="log_email")
        show_pass_log = st.checkbox("Show password", key="show_log")
        log_pass = st.text_input("Password", type="default" if show_pass_log else "password", key="log_pass")
        
        if st.button("Login", type="primary", use_container_width=True):
            try:
                res = APIClient.login(log_email, log_pass)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.logged_in = True
                    st.session_state.user_email = log_email.lower()
                    # Bug 4 Fix: persist JWT token for authenticated API calls
                    st.session_state.auth_token = data.get("access_token", "")
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Login failed"))
            except Exception as e:
                st.error("Backend server is offline. Make sure `uvicorn main:app --reload` is running.")

    with tab2:
        st.subheader("Create a new account")
        sign_email = st.text_input("Email", key="sign_email")
        show_pass_sign = st.checkbox("Show password", key="show_sign")
        sign_pass = st.text_input("Password", type="default" if show_pass_sign else "password", key="sign_pass")
        
        if st.button("Sign Up", type="primary", use_container_width=True):
            try:
                res = APIClient.signup(sign_email, sign_pass)
                # Bug 10 Fix: signup now returns 201 Created
                if res.status_code == 201:
                    st.success("Account created successfully! Please go to the Login tab.")
                else:
                    st.error(res.json().get("detail", "Sign up failed"))
            except Exception:
                st.error("Backend server is offline.")
                
    with tab3:
        st.subheader("Reset Password")
        st.info("For this prototype, enter your email and a new password to reset it directly.")
        reset_email = st.text_input("Account Email", key="reset_email")
        show_pass_reset = st.checkbox("Show new password", key="show_reset")
        new_pass = st.text_input("New Password", type="default" if show_pass_reset else "password", key="reset_pass")
        
        if st.button("Reset Password", type="primary", use_container_width=True):
            try:
                res = APIClient.reset_password(reset_email, new_pass)
                if res.status_code == 200:
                    st.success("Password reset! You can now log in.")
                else:
                    st.error(res.json().get("detail", "Email not found"))
            except Exception:
                st.error("Backend server is offline.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE & HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 My Profile":
    st.markdown('<p class="main-title">👤 My Profile & History</p>', unsafe_allow_html=True)
    st.write(f"**Account Email:** {st.session_state.user_email}")
    st.divider()
    
    st.markdown('<p class="section-header">📊 Your Saved Resume Scores</p>', unsafe_allow_html=True)
    try:
        history = APIClient.get_history(st.session_state.user_email)
        if not history:
            st.info("You haven't saved any resume scores yet. Go to the 'IT Resume Scorer' to get started!")
        else:
            df = pd.DataFrame(history)
            df = df[["target_domain", "probability", "experience", "education", "skill_gap"]]
            df.columns = ["Target Role", "ATS Pass Probability (%)", "Experience (Yrs)", "Education", "Skill Gap (%)"]
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        # Bug 11 Fix: surface the real reason so user knows what to do
        err_msg = str(e)
        if "401" in err_msg:
            st.error("⚠️ Session expired. Please log out and log back in.")
        elif "503" in err_msg:
            st.error("⚠️ Database is temporarily unavailable. Try again later.")
        else:
            st.error(f"Could not load history: {err_msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Home":
    st.markdown('<p class="main-title">💻 AI-Powered IT Career System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Align your tech stack · Close your skill gaps · Win the shortlist</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("IT Domains", f"{len(DOMAIN_PROFILES)}", "covered")
    with c2: st.metric("Tech Skills", f"{len(processor.SKILL_KEYWORDS)}+", "tracked")
    with c3: st.metric("ML Model", "Random Forest", "Shortlist Predictor")
    with c4: st.metric("Explainability", "SHAP", "visualizations")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🎯 System Capabilities")
        st.markdown("""
- **Tech Stack Matching** — Finds your ideal IT role using set-based similarity.
- **Skill Gap Analysis** — Identifies missing frameworks/languages and ranks them.
- **Resume Scoring** — Predicts the likelihood your tech resume passes an ATS/HR screening.
- **AI Explainability** — Uses SHAP to break down exactly which metrics impacted your score.
- **Developer Roadmap** — Generates a week-by-week learning plan to close your tech gap.
        """)

    with col_r:
        st.markdown("### 🏗 Supported IT Tracks")
        for domain, profile in DOMAIN_PROFILES.items():
            st.markdown(f"**{domain}** — *(e.g., {', '.join(profile['required_skills'][:3])})*")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CAREER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Tech Skill Analysis":
    st.markdown('<p class="main-title">🔍 Tech Path Finder</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter your tech stack to discover your best IT path and what to learn next</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Step 1 — Enter Your Tech Stack</p>', unsafe_allow_html=True)
    input_method = st.radio("Input method:", ["Type skills manually", "Upload resume PDF"], horizontal=True)

    user_skills = []
    
    if input_method == "Upload resume PDF":
        uploaded = st.file_uploader("Upload your tech resume (PDF):", type=["pdf"])
        if uploaded:
            with st.spinner("Extracting stack from PDF..."):
                import tempfile
                # Bug 3 Fix: try/finally guarantees temp file is deleted even if processing crashes
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    result = processor.process_resume(tmp_path, is_pdf=True)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                st.session_state.memory_skills = ", ".join([s.title() for s in result["skills"]])
                st.session_state.memory_raw_text = result["raw_text"]

                text_lower = result["raw_text"].lower()
                # Bug 6 Fix: extract year ranges instead of using word-count / 300 heuristic
                year_matches = re.findall(r"\b(19|20)\d{2}\b", text_lower)
                unique_years = sorted(set(int(y) for y in year_matches))
                exp_from_years = float(unique_years[-1] - unique_years[0]) if len(unique_years) >= 2 else 0.0
                st.session_state.memory_exp = min(max(exp_from_years, 0.0), 10.0)
                if "senior" in text_lower: st.session_state.memory_exp = max(st.session_state.memory_exp, 5.0)
                if any(w in text_lower for w in ["phd", "master", "m.tech"]): st.session_state.memory_edu = "Master's / PhD"
                elif any(w in text_lower for w in ["bachelor", "b.tech", "bsc"]): st.session_state.memory_edu = "Bachelor's"
                else: st.session_state.memory_edu = "No degree"
                st.session_state.memory_certs = min(len(re.findall(r"\bcertif\w+", text_lower)), 10)
                # Fix: count actual project entries using "Title | Technologies" separators
                # and GitHub references — far more accurate than counting the word "project"
                proj_pipes  = len(re.findall(r'\|\s*[a-z]', text_lower))  # e.g. "App Name | Flutter, Firebase"
                proj_github = len(re.findall(r'github\s*[:\.]', text_lower))  # e.g. "GitHub:SubTrack"
                proj_keyword = max(len(re.findall(r'\bprojects?\b', text_lower)) - 1, 0)  # subtract section header
                proj_count  = max(proj_pipes, proj_github, proj_keyword, 1)
                st.session_state.memory_proj = min(proj_count, 15)

                st.success("✅ Extraction complete! Skills saved to memory.")

    skills_input = st.text_area(
        "Your Skills (comma-separated):",
        value=st.session_state.memory_skills,
        placeholder="python, react, docker, aws, sql...",
        height=100,
    )
    
    if skills_input != st.session_state.memory_skills:
        st.session_state.memory_skills = skills_input
        st.session_state.manual_extracted_skills = []
        st.session_state.manual_skills_extracted = False

    if input_method == "Type skills manually":
        if st.button("Extract Skill", type="primary", use_container_width=True):
            if not st.session_state.memory_skills.strip():
                st.warning("Please enter your skills first.")
                st.session_state.manual_extracted_skills = []
                st.session_state.manual_skills_extracted = False
            else:
                extracted = processor.skills_from_list(st.session_state.memory_skills)
                known_map = {k.lower(): k for k in processor.SKILL_KEYWORDS}
                st.session_state.manual_extracted_skills = [known_map[s] if s in known_map else s.title() for s in extracted]
                st.session_state.manual_skills_extracted = True
                st.success("✅ Skills extracted from manual input.")

        if st.session_state.manual_skills_extracted:
            user_skills = st.session_state.manual_extracted_skills
    elif st.session_state.memory_skills.strip():
        user_skills = processor.skills_from_list(st.session_state.memory_skills)
        known_map = {k.lower(): k for k in processor.SKILL_KEYWORDS}
        user_skills = [known_map[s] if s in known_map else s.title() for s in user_skills]

    if user_skills:
        st.markdown("**Detected Tech Stack:**")
        tags_html = "".join(f'<span class="skill-tag-green">{s}</span>' for s in user_skills)
        st.markdown(tags_html, unsafe_allow_html=True)
        st.divider()

        st.markdown('<p class="section-header">Step 2 — IT Domain Match</p>', unsafe_allow_html=True)
        domain_result = matcher.recommend(user_skills)
        rankings_df = matcher.get_all_rankings_df(user_skills)

        col1, col2 = st.columns([1, 2])
        with col1:
            top = domain_result["top_domain"]
            fit = domain_result["fit_score"]
            score_class = "score-high" if fit >= 70 else ("score-mid" if fit >= 40 else "score-low")
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-card-title">🏆 Best IT Match</div>
    <div class="metric-card-val">{top}</div>
    <div class="{score_class}">{fit}%</div>
    <div class="metric-card-title">Stack Alignment Score</div>
</div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Matched Stack:** {', '.join(domain_result['user_skills_matched']) or 'None'}")

        with col2:
            fig_rank = px.bar(
                rankings_df, x="Fit Score (%)", y="Domain",
                orientation="h", color="Fit Score (%)",
                color_continuous_scale="Tealgrn",
            )
            fig_rank.update_layout(
                height=320, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            fig_rank.update_layout(font_color=st.get_option("theme.textColor") or "gray")
            st.plotly_chart(fig_rank, use_container_width=True)

        st.divider()

        st.markdown('<p class="section-header">Step 3 — Technical Gap Analysis</p>', unsafe_allow_html=True)
        selected_domain = st.selectbox(
            "Analyze missing tech for:",
            options=list(DOMAIN_PROFILES.keys()),
            index=list(DOMAIN_PROFILES.keys()).index(top),
        )

        gap_report = analyzer.analyze(user_skills, selected_domain)
        col_a, col_b, col_c = st.columns(3)
        with col_a: st.metric("Stack Coverage", f"{gap_report['coverage_score']}%")
        with col_b: st.metric("Technical Gap", f"{gap_report['gap_percentage']}%")
        with col_c: st.metric("Missing Core Tech", len(gap_report["missing_required"]))

        col_m, col_miss = st.columns(2)
        with col_m:
            st.markdown("**✅ Core Tech You Have**")
            if gap_report["matched_skills"]:
                st.markdown("".join(f'<span class="skill-tag-green">{s}</span>' for s in gap_report["matched_skills"]), unsafe_allow_html=True)
            else:
                st.caption("None matched yet.")

        with col_miss:
            st.markdown("**❌ Missing Core Tech**")
            if gap_report["missing_required"]:
                st.markdown("".join(f'<span class="skill-tag-red">{s}</span>' for s in gap_report["missing_required"]), unsafe_allow_html=True)
            else:
                st.success("You hit all core requirements! 🎉")

        st.divider()
        st.markdown('<p class="section-header">📅 Developer Learning Roadmap</p>', unsafe_allow_html=True)
        roadmap = analyzer.roadmap(user_skills, selected_domain)
        if roadmap:
            st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)
        else:
            st.success("🎉 You are technically ready for this role!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RESUME SCORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 IT Resume Scorer":
    st.markdown('<p class="main-title">📄 Tech Resume Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict the probability your resume clears ATS for IT roles</p>', unsafe_allow_html=True)

    if not predictor.is_fitted:
        st.error("⚠️ Model not trained yet. Run `train_models.ipynb` to generate `.pkl` files.")
        st.stop()

    if st.session_state.memory_skills:
        st.info("💡 Candidate Profile pre-filled from your Session Memory.")

    st.markdown('<p class="section-header">Candidate Profile</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        skills_input = st.text_area("Tech Stack (comma-separated):", value=st.session_state.memory_skills, height=90)
        
        if skills_input != st.session_state.memory_skills:
            st.session_state.memory_skills = skills_input
            
        dynamic_domain_index = 0
        if st.session_state.memory_skills:
            temp_skills = processor.skills_from_list(st.session_state.memory_skills)
            if temp_skills:
                suggested_domain = matcher.recommend(temp_skills)["top_domain"]
                dynamic_domain_index = list(DOMAIN_PROFILES.keys()).index(suggested_domain)

        target_domain = st.selectbox("Target IT Role:", list(DOMAIN_PROFILES.keys()), index=dynamic_domain_index)

    with col2:
        # BUG 1 FIX: Using float(round(...)) to prevent slider crash on weird decimals
        experience = st.slider("Years of IT Experience", 0.0, 10.0, float(round(st.session_state.memory_exp, 1)))
        projects = st.slider("Portfolio Projects", 0, 15, int(st.session_state.memory_proj))
        
        edu_options = ["No degree", "Bachelor's", "Master's / PhD"]
        safe_edu = st.session_state.memory_edu if st.session_state.memory_edu in edu_options else "Bachelor's"
        education = st.selectbox("Highest Education", edu_options, index=edu_options.index(safe_edu))
        certs = st.slider("Tech Certifications", 0, 10, int(st.session_state.memory_certs))
        
        st.session_state.memory_exp = experience
        st.session_state.memory_proj = projects
        st.session_state.memory_edu = education
        st.session_state.memory_certs = certs

    edu_score = {"No degree": 0, "Bachelor's": 1, "Master's / PhD": 2}[education]

    if st.button("🔮 Predict Shortlisting Probability", type="primary", use_container_width=True):
        if not st.session_state.memory_skills.strip():
            st.warning("Please enter your tech stack first.")
        else:
            user_skills = processor.skills_from_list(st.session_state.memory_skills)
            with st.spinner("Running ML model..."):
                domain_result = matcher.recommend(user_skills)
                gap_report = analyzer.analyze(user_skills, target_domain)

                full_context_text = st.session_state.memory_raw_text + " " + st.session_state.memory_skills
                features = ResumePredictor.engineer_features(
                    user_skills=user_skills, domain=target_domain, experience_years=experience,
                    project_count=projects, education_score=edu_score, certification_count=certs,
                    full_text=full_context_text, 
                    # BUG 2 FIX: Ensure the ML gets the score for the dropdown target_domain, not 100%
                    domain_fit_score=gap_report["coverage_score"],
                    gap_percentage=gap_report["gap_percentage"],
                )

                prob, label = predictor.predict_proba(features)
                # Realistic capping is now handled inside predict_proba via weighted blending

            st.divider()
            col_score, col_gauge = st.columns([1, 2])
            score_class = "score-high" if prob >= 70 else ("score-mid" if prob >= 45 else "score-low")
            
            with col_score:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-card-title">ATS Pass Probability</div>
    <div class="{score_class}">{prob}%</div>
    <div class="metric-card-title">{label}</div>
</div>
                """, unsafe_allow_html=True)

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00c6ff"}},
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=20, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
                fig_gauge.update_layout(font_color=st.get_option("theme.textColor") or "gray")
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()
            st.markdown('<p class="section-header">🧠 Why you got this score (SHAP Analysis)</p>', unsafe_allow_html=True)
            
            explainer_obj = Explainer(predictor.model, predictor.scaler, predictor.FEATURE_COLUMNS)
            explanations = explainer_obj.explain_text(features, prob)

            for exp in explanations[:5]:
                icon = "✅" if exp["direction"] == "positive" else "❌"
                st.markdown(f"{icon} **{exp['feature'].replace('_',' ').title()}** (`{exp['value']}`) — {exp['message']}")

            col_wf, col_radar = st.columns(2)
            with col_wf:
                st.caption("Feature Impact (Simulated Waterfall)")
                fig_wf = explainer_obj.waterfall_plot(features, prob)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)
            with col_radar:
                st.caption("Candidate vs Ideal IT Profile")
                fig_radar = explainer_obj.radar_chart(features)
                st.pyplot(fig_radar, use_container_width=True)
                plt.close(fig_radar)

            # --- STORE PAYLOAD FOR SAVE BUTTON (outside predict block) ---
            st.session_state.last_prediction_payload = {
                "user_email": st.session_state.user_email,
                "skills": user_skills,
                "target_domain": target_domain,
                "experience": experience,
                "education": education,
                "probability": prob,
                "label": label,
                "skill_gap": gap_report["gap_percentage"],
            }
            st.session_state.last_save_status = None  # reset status on new prediction

    # ── Save button lives OUTSIDE the Predict block so it survives Streamlit re-runs ──
    # (clicking any button re-runs the script; buttons inside other button blocks
    #  disappear on re-run because the outer button is no longer "pressed")
    if st.session_state.last_prediction_payload:
        st.divider()
        if not st.session_state.get("auth_token"):
            st.info("💡 Session may have expired — log out and back in to enable saving.")
        else:
            if st.session_state.last_save_status == "saved":
                st.success("✅ Already saved to your Profile History!")
            elif st.button("💾 Save to My Profile History", type="primary", use_container_width=True):
                try:
                    with st.spinner("Saving..."):
                        response = APIClient.save_record(st.session_state.last_prediction_payload)
                    if response.status_code == 201:
                        st.session_state.last_save_status = "saved"
                        st.success(f"✅ Analysis saved! (DB ID: {response.json().get('id')})")
                        st.rerun()  # refresh so button switches to ✅ message
                    elif response.status_code == 401:
                        st.warning("⚠️ Session expired. Please log out and log back in.")
                    else:
                        detail = response.json().get('detail', response.text)
                        st.warning(f"⚠️ Could not save ({response.status_code}): {detail}")
                except Exception as e:
                    st.warning(f"⚠️ Could not reach the backend: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DOMAIN EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Tech Domain Explorer":
    st.markdown('<p class="main-title">📊 IT Domain Explorer</p>', unsafe_allow_html=True)
    
    selected = st.selectbox("Select an IT domain:", list(DOMAIN_PROFILES.keys()))
    profile = DOMAIN_PROFILES[selected]

    col1, col2, col3 = st.columns(3)
    col1.metric("Core Tech Required", len(profile["required_skills"]))
    col2.metric("Bonus Tech", len(profile["nice_to_have"]))
    col3.metric("Avg. Experience", f"{profile['avg_experience_years']} yrs")

    st.markdown(f"> *{profile['description']}*")
    st.divider()

    all_skills = {}
    for d, p in DOMAIN_PROFILES.items():
        for skill in p["required_skills"]:
            all_skills[skill] = all_skills.get(skill, 0) + 1

    freq_df = pd.DataFrame(list(all_skills.items()), columns=["Skill", "Roles Demanding It"]).sort_values("Roles Demanding It", ascending=False).head(15)
    
    fig_freq = px.bar(freq_df, x="Roles Demanding It", y="Skill", orientation="h", title="Most Versatile IT Skills Across All Domains")
    fig_freq.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_freq.update_layout(font_color=st.get_option("theme.textColor") or "gray")
    st.plotly_chart(fig_freq, use_container_width=True)