import pandas as pd

# IT-Specific Career Domains
DOMAIN_PROFILES = {
    "AI & Machine Learning": {
        "description": "Building intelligent systems, LLMs/SLMs, and predictive models.",
        "required_skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "sql", "pandas", "numpy"],
        "nice_to_have": ["nlp", "computer vision", "docker", "aws", "scikit-learn", "llms"],
        "avg_experience_years": 3.0
    },
    "Mobile App Development": {
        "description": "Designing and building cross-platform or native mobile applications.",
        "required_skills": ["flutter", "dart", "react native", "swift", "kotlin", "java", "api integration", "git"],
        "nice_to_have": ["firebase", "ui/ux design", "sqlite", "ci/cd", "state management"],
        "avg_experience_years": 2.0
    },
    "Backend Engineering": {
        "description": "Developing robust server-side logic, APIs, and database architectures.",
        "required_skills": ["python", "java", "node.js", "sql", "rest apis", "git", "linux"],
        "nice_to_have": ["django", "fastapi", "docker", "kubernetes", "postgresql", "redis", "graphql"],
        "avg_experience_years": 2.5
    },
    "Frontend Engineering": {
        "description": "Creating responsive, interactive, and accessible user interfaces.",
        "required_skills": ["html", "css", "javascript", "react", "vue", "git"],
        "nice_to_have": ["typescript", "tailwind css", "next.js", "redux", "figma"],
        "avg_experience_years": 2.0
    },
    "Cloud & DevOps": {
        "description": "Automating infrastructure, deployments, and managing cloud environments.",
        "required_skills": ["linux", "bash", "aws", "docker", "kubernetes", "ci/cd", "git"],
        "nice_to_have": ["terraform", "azure", "gcp", "ansible", "jenkins", "python"],
        "avg_experience_years": 3.5
    },
    "Data Engineering": {
        "description": "Building data pipelines, warehouses, and optimizing data retrieval.",
        "required_skills": ["sql", "python", "etl", "spark", "hadoop", "data modeling"],
        "nice_to_have": ["kafka", "airflow", "aws redshift", "snowflake", "scala"],
        "avg_experience_years": 3.0
    },
    "Cybersecurity": {
        "description": "Protecting systems, networks, and data from digital attacks.",
        "required_skills": ["linux", "networking", "python", "penetration testing", "firewalls", "cryptography"],
        "nice_to_have": ["wireshark", "aws security", "siem", "ethical hacking", "bash"],
        "avg_experience_years": 4.0
    }
}

class DomainMatcher:
    def recommend(self, user_skills):
        user_skills_set = set([s.lower() for s in user_skills])
        best_domain = None
        best_score = -1
        matched_skills_for_best = []
        
        for domain, profile in DOMAIN_PROFILES.items():
            req_skills = set([s.lower() for s in profile["required_skills"]])
            match_count = len(user_skills_set.intersection(req_skills))
            score = (match_count / len(req_skills)) * 100 if req_skills else 0
            
            if score > best_score:
                best_score = score
                best_domain = domain
                matched_skills_for_best = list(user_skills_set.intersection(req_skills))
                
        return {
            "top_domain": best_domain if best_domain else list(DOMAIN_PROFILES.keys())[0],
            "fit_score": round(best_score, 1),
            "domain_info": DOMAIN_PROFILES.get(best_domain, DOMAIN_PROFILES[list(DOMAIN_PROFILES.keys())[0]]),
            "user_skills_matched": [s.title() for s in matched_skills_for_best]
        }

    def get_all_rankings_df(self, user_skills):
        user_skills_set = set([s.lower() for s in user_skills])
        data = []
        for domain, profile in DOMAIN_PROFILES.items():
            req_skills = set([s.lower() for s in profile["required_skills"]])
            score = (len(user_skills_set.intersection(req_skills)) / len(req_skills)) * 100 if req_skills else 0
            data.append({"Domain": domain, "Fit Score (%)": round(score, 1)})
            
        df = pd.DataFrame(data).sort_values(by="Fit Score (%)", ascending=True)
        return df