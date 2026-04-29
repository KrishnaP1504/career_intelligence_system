"""
jd_analyzer.py — Job Description Analyzer
──────────────────────────────────────────────────────────────────────────────
Compares a pasted Job Description against the user's profile to surface:
  • Exact keyword matches & misses
  • Skill-gap percentage
  • Actionable resume-edit suggestions
──────────────────────────────────────────────────────────────────────────────
"""

import re
from collections import Counter
from .nlp_processor import NLPProcessor
from .domain_matcher import DOMAIN_PROFILES


class JDAnalyzer:
    """Analyses a job description against a candidate's existing skill set."""

    # Common JD "action" / "requirement" phrases (lowercased)
    EXPERIENCE_PATTERNS = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(?:minimum|at\s*least)\s*(\d+)\s*(?:years?|yrs?)",
    ]

    EDUCATION_KEYWORDS = {
        "phd": "PhD",
        "ph.d": "PhD",
        "master": "Master's",
        "m.tech": "Master's",
        "m.s.": "Master's",
        "bachelor": "Bachelor's",
        "b.tech": "Bachelor's",
        "b.s.": "Bachelor's",
        "b.sc": "Bachelor's",
    }

    # Additional JD-specific buzzwords that are NOT in the skill set but are
    # commonly checked by ATS systems
    ATS_BUZZWORDS = {
        "agile", "scrum", "ci/cd", "microservices", "rest apis", "graphql",
        "data structures", "algorithms", "system design", "distributed systems",
        "unit testing", "integration testing", "code review", "mentoring",
        "cross-functional", "stakeholder management", "problem solving",
        "communication", "team player", "leadership", "project management",
        "version control", "git", "github", "gitlab",
    }

    def __init__(self, nlp_processor: NLPProcessor | None = None):
        self.processor = nlp_processor

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Collapse whitespace and strip."""
        return re.sub(r"\s+", " ", text).strip()

    def _extract_skills_from_text(self, text: str) -> list[str]:
        """Extract known tech skills from arbitrary text."""
        text_lower = text.lower()
        found: set[str] = set()

        # Single-token match
        tokens = re.findall(r"[\w#+./-]+", text_lower)
        for token in tokens:
            if token in NLPProcessor.SKILL_KEYWORDS:
                found.add(token)

        # Multi-word skill match
        for skill in NLPProcessor.SKILL_KEYWORDS:
            if " " in skill and skill in text_lower:
                found.add(skill)

        return sorted(found)

    def _extract_experience_years(self, text: str) -> int | None:
        """Try to pull minimum years-of-experience from a JD."""
        text_lower = text.lower()
        for pattern in self.EXPERIENCE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        return None

    def _extract_education(self, text: str) -> str | None:
        """Try to detect required education level from the JD."""
        text_lower = text.lower()
        for keyword, level in self.EDUCATION_KEYWORDS.items():
            if keyword in text_lower:
                return level
        return None

    def _extract_ats_buzzwords(self, text: str) -> list[str]:
        """Find soft/process buzzwords that ATS scanners look for."""
        text_lower = text.lower()
        found = set()
        for bw in self.ATS_BUZZWORDS:
            if bw in text_lower:
                found.add(bw)
        return sorted(found)

    # ── main analysis ─────────────────────────────────────────────────────────

    def analyze(
        self,
        jd_text: str,
        user_skills: list[str],
        user_experience: float = 0.0,
        user_education: str = "Bachelor's",
    ) -> dict:
        """
        Compare a Job Description against the user's profile.

        Returns
        -------
        dict  with keys:
            jd_skills           – list of skills found in the JD
            matched_skills      – skills the user already has
            missing_skills      – skills the user is lacking
            match_percentage    – overall keyword-match %
            ats_buzzwords_in_jd – soft/process keywords the JD mentions
            ats_buzzwords_user_has  – of those, which the user could claim
            experience_required – years (int) or None
            experience_met      – bool
            education_required  – str or None
            education_met       – bool
            suggestions         – list of actionable resume-edit strings
            priority_skills     – top-5 missing skills ranked by domain frequency
        """
        jd_text = self._clean(jd_text)
        jd_skills = self._extract_skills_from_text(jd_text)
        user_set = set(s.lower() for s in user_skills)

        matched = sorted(set(jd_skills) & user_set)
        missing = sorted(set(jd_skills) - user_set)

        total_jd = len(jd_skills) if jd_skills else 1
        match_pct = round(len(matched) / total_jd * 100, 1)

        # Experience check
        exp_required = self._extract_experience_years(jd_text)
        exp_met = True
        if exp_required is not None:
            exp_met = user_experience >= exp_required

        # Education check
        edu_required = self._extract_education(jd_text)
        edu_met = True
        edu_hierarchy = {"No degree": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3, "Master's / PhD": 2}
        if edu_required:
            user_lvl = edu_hierarchy.get(user_education, 1)
            req_lvl = edu_hierarchy.get(edu_required, 1)
            edu_met = user_lvl >= req_lvl

        # ATS buzzwords
        jd_buzzwords = self._extract_ats_buzzwords(jd_text)
        user_buzzwords = [bw for bw in jd_buzzwords if bw in user_set]

        # Priority ranking: skills that appear in MOST domain profiles → more versatile
        skill_frequency: Counter = Counter()
        for profile in DOMAIN_PROFILES.values():
            for s in profile["required_skills"] + profile["nice_to_have"]:
                skill_frequency[s.lower()] += 1
        priority = sorted(missing, key=lambda s: skill_frequency.get(s, 0), reverse=True)[:5]

        # ── build suggestions ──────────────────────────────────────────────
        suggestions: list[str] = []

        if missing:
            top3 = missing[:3]
            suggestions.append(
                f"🔑 **Add missing keywords**: Incorporate {', '.join(s.title() for s in top3)} "
                f"into your Skills section or project descriptions."
            )

        if missing:
            suggestions.append(
                "📝 **Mirror JD language**: Use the exact phrasing from the job description — "
                "ATS systems do literal string matching."
            )

        if not exp_met and exp_required:
            suggestions.append(
                f"⏳ **Experience gap**: The JD asks for {exp_required}+ years. "
                f"Highlight freelance work, internships, and academic projects to close this gap."
            )

        if not edu_met and edu_required:
            suggestions.append(
                f"🎓 **Education**: The JD prefers a {edu_required} degree. "
                f"Emphasize relevant certifications or coursework."
            )

        missing_buzzwords = [bw for bw in jd_buzzwords if bw not in user_set]
        if missing_buzzwords:
            suggestions.append(
                f"💬 **Soft/process keywords**: Add {', '.join(bw.title() for bw in missing_buzzwords[:3])} "
                f"to your resume summary or experience bullets."
            )

        if match_pct >= 80:
            suggestions.append(
                "✅ **Strong match!** Fine-tune bullet points to echo the JD's exact wording."
            )
        elif match_pct >= 50:
            suggestions.append(
                "⚡ **Decent alignment** — focus your next project on the top missing skill to push above 80%."
            )
        else:
            suggestions.append(
                "🚨 **Low match**: Consider whether this role truly fits. If yes, "
                "build 1–2 portfolio projects using the missing tech stack."
            )

        # Suggest quantifying achievements
        suggestions.append(
            "📊 **Quantify achievements**: Replace vague statements with metrics — "
            "e.g. \"Reduced API latency by 40%\" or \"Served 10K+ daily users\"."
        )

        return {
            "jd_skills": jd_skills,
            "matched_skills": matched,
            "missing_skills": missing,
            "match_percentage": match_pct,
            "ats_buzzwords_in_jd": jd_buzzwords,
            "ats_buzzwords_user_has": user_buzzwords,
            "experience_required": exp_required,
            "experience_met": exp_met,
            "education_required": edu_required,
            "education_met": edu_met,
            "suggestions": suggestions,
            "priority_skills": priority,
        }
