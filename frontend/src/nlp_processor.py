import re
import spacy
import pdfplumber

class NLPProcessor:
    # Bug 9 Fix: Synchronized with all skills appearing in DOMAIN_PROFILES
    SKILL_KEYWORDS = {
        # Languages
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "scala",
        # Frontend frameworks
        "react", "angular", "vue", "next.js", "redux", "tailwind css", "figma",
        # Backend frameworks
        "node.js", "django", "flask", "fastapi", "graphql", "rest apis",
        # Mobile
        "flutter", "dart", "react native", "swift", "kotlin", "sqlite",
        "api integration", "firebase", "state management", "ui/ux design",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
        "linux", "bash", "ci/cd", "agile", "scrum",
        "aws security", "aws redshift",
        # ML / Data Science
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "llms", "data analysis",
        # Data Engineering
        "etl", "spark", "hadoop", "data modeling", "kafka", "airflow", "snowflake",
        # Security
        "cybersecurity", "networking", "penetration testing", "firewalls",
        "cryptography", "wireshark", "siem", "ethical hacking",
        # Version control / collaboration
        "git", "github", "gitlab",
    }

    def __init__(self):
        # Bug 5 Fix: wrap the fallback download in a try/except with a clear RuntimeError
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    timeout=120,
                )
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as download_err:
                raise RuntimeError(
                    f"spaCy model 'en_core_web_sm' not found and auto-download failed: {download_err}. "
                    "Run: python -m spacy download en_core_web_sm"
                )

        self.vectorizer = None

    def load_vectorizer(self, path):
        import pickle
        try:
            with open(path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            pass  # Vectorizer only needed for advanced TF-IDF features

    def process_resume(self, file_path, is_pdf=True):
        text = ""
        if is_pdf:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except Exception as e:
                print(f"Error reading PDF: {e}")

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract skills using spaCy token matching
        doc = self.nlp(text.lower())
        found_skills = set()

        for token in doc:
            if token.text in self.SKILL_KEYWORDS:
                found_skills.add(token.text)

        # N-gram matching for multi-word skills
        text_lower = text.lower()
        for skill in self.SKILL_KEYWORDS:
            if " " in skill and skill in text_lower:
                found_skills.add(skill)

        return {
            "raw_text": text,
            "skills": [s.title() for s in found_skills],
        }

    def skills_from_list(self, text_input):
        # Bug 7 Fix: only return tokens that exist in SKILL_KEYWORDS
        # Prevents free-text phrases from being saved to MongoDB as "skills"
        raw = [s.strip().lower() for s in text_input.split(",")]
        return [s for s in raw if s and s in self.SKILL_KEYWORDS]