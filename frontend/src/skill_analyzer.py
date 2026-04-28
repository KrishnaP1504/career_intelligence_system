import pandas as pd
from .domain_matcher import DOMAIN_PROFILES

class SkillAnalyzer:
    def analyze(self, user_skills, domain):
        user_set = set([s.lower() for s in user_skills])
        profile = DOMAIN_PROFILES.get(domain, list(DOMAIN_PROFILES.values())[0])
        
        req_set = set([s.lower() for s in profile["required_skills"]])
        nice_set = set([s.lower() for s in profile["nice_to_have"]])
        
        matched_req = user_set.intersection(req_set)
        missing_req = req_set - user_set
        missing_nice = nice_set - user_set
        
        coverage = (len(matched_req) / len(req_set)) * 100 if req_set else 0
        gap = 100 - coverage
        
        # Calculate improvement if they learn the top 2 missing skills
        skills_to_add = list(missing_req)[:2]
        new_coverage = ((len(matched_req) + len(skills_to_add)) / len(req_set)) * 100 if req_set else 0
        improvement = new_coverage - coverage
        
        return {
            "coverage_score": round(coverage, 1),
            "gap_percentage": round(gap, 1),
            "matched_skills": [s.title() for s in matched_req],
            "missing_required": [s.title() for s in missing_req],
            "missing_nice_to_have": [s.title() for s in missing_nice],
            "improvement_impact": {
                "skills_to_add": [s.title() for s in skills_to_add],
                "improvement_pct": round(improvement, 1)
            }
        }

    def compare_domains(self, user_skills):
        user_set = set([s.lower() for s in user_skills])
        data = []
        for d, p in DOMAIN_PROFILES.items():
            req_set = set([s.lower() for s in p["required_skills"]])
            cov = (len(user_set.intersection(req_set)) / len(req_set)) * 100 if req_set else 0
            data.append({"Domain": d, "Coverage (%)": round(cov, 1), "Gap (%)": round(100 - cov, 1)})
        return pd.DataFrame(data).sort_values(by="Gap (%)", ascending=True)

    def roadmap(self, user_skills, domain):
        analysis = self.analyze(user_skills, domain)
        missing = analysis["missing_required"]
        
        roadmap_data = []
        for i, skill in enumerate(missing):
            week = f"Week {i*2+1}–{i*2+2}"  # Bug 8 Fix: non-overlapping ranges
            roadmap_data.append({
                "Timeline": week,
                "Skill to Learn": skill,
                "Action Item": f"Complete foundational tutorial and build 1 mini-project using {skill}"
            })
            
        return roadmap_data