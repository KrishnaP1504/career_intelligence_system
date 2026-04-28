import pickle
import numpy as np
import pandas as pd


class ResumePredictor:
    # Must match the columns generated in train_models.ipynb exactly
    FEATURE_COLUMNS = [
        'experience_years', 'project_count', 'education_score',
        'certification_count', 'domain_fit_score', 'gap_percentage'
    ]

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def load_model(self, model_path, scaler_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
        except FileNotFoundError:
            self.is_fitted = False

    @staticmethod
    def engineer_features(user_skills, domain, experience_years, project_count,
                          education_score, certification_count, full_text,
                          domain_fit_score, gap_percentage):
        return {
            'experience_years':    float(experience_years),
            'project_count':       int(project_count),
            'education_score':     int(education_score),
            'certification_count': int(certification_count),
            'domain_fit_score':    float(domain_fit_score),
            'gap_percentage':      float(gap_percentage),
        }

    def predict_proba(self, features_dict):
        if not self.is_fitted:
            return 0.0, "Model Offline"

        # ── Raw ML prediction ────────────────────────────────────────────────
        # Fix: use a named DataFrame so it matches how the StandardScaler was
        # originally fitted — eliminates the sklearn "feature names" UserWarning
        feature_df = pd.DataFrame([features_dict], columns=self.FEATURE_COLUMNS)
        scaled     = self.scaler.transform(feature_df)
        raw_prob   = self.model.predict_proba(scaled)[0][1] * 100

        # ── Transparent weighted scoring formula ─────────────────────────────
        # The trained RandomForest is overconfident (always ≥99%).
        # We blend 25% ML output with 75% weighted formula so the score
        # actually changes meaningfully when the user tweaks their profile.
        #
        # Max points per component  →  total = 100
        #   Experience    : 25 pts  (≥6 yrs = full)
        #   Projects      : 20 pts  (≥4 projects = full)
        #   Education     : 15 pts  (Master's/PhD = full)
        #   Certifications: 10 pts  (≥4 certs = full)
        #   Domain fit    : 20 pts  (100% skill match = full)
        #   Skill coverage: 10 pts  (0% gap = full)

        exp   = features_dict['experience_years']
        proj  = features_dict['project_count']
        edu   = features_dict['education_score']
        certs = features_dict['certification_count']
        fit   = features_dict['domain_fit_score']
        gap   = features_dict['gap_percentage']

        weighted_score = (
            min(exp  / 6.0, 1.0) * 25 +
            min(proj / 4.0, 1.0) * 20 +
            (edu / 2.0)          * 15 +
            min(certs / 4.0, 1.0)* 10 +
            (fit / 100.0)        * 20 +
            ((100.0 - gap) / 100.0) * 10
        )

        # Blend and clamp to realistic range [5, 97]
        prob = round(0.25 * raw_prob + 0.75 * weighted_score, 1)
        prob = float(min(max(prob, 5.0), 97.0))

        if prob >= 70:
            label = "Highly likely to be shortlisted"
        elif prob >= 45:
            label = "Average chance, needs optimization"
        else:
            label = "Low chance, significant skill gaps detected"

        return prob, label

    def feature_importance_df(self):
        if not self.is_fitted:
            return pd.DataFrame()

        importances = self.model.feature_importances_ * 100
        df = pd.DataFrame({
            "Feature":        [c.replace('_', ' ').title() for c in self.FEATURE_COLUMNS],
            "Importance (%)": importances,
        })
        return df.sort_values(by="Importance (%)", ascending=True)