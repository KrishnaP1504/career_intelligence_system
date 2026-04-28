import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Explainer:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.safe_names = [f.replace('_', ' ').title() for f in feature_names]

    def explain_text(self, features_dict, base_prob):
        avg_features = {
            'experience_years': 3.0,
            'project_count': 5,
            'education_score': 1,
            'certification_count': 1,
            'domain_fit_score': 65.0,
            'gap_percentage': 35.0,
        }

        explanations = []
        for feature in self.feature_names:
            val = features_dict[feature]
            avg = avg_features[feature]

            impact = 0.0
            direction = "neutral"
            message = ""

            if feature == 'gap_percentage':
                if val < avg:
                    direction = "positive"
                    message = "Your skill gap is lower than average."
                    impact = abs(val - avg) * 0.5
                else:
                    direction = "negative"
                    message = "Missing required skills is hurting your score."
                    impact = -abs(val - avg) * 0.5
            elif val > avg:
                direction = "positive"
                message = "Above average for this metric."
                impact = abs(val - avg) * (1.5 if feature == 'domain_fit_score' else 1.0)
            elif val < avg:
                direction = "negative"
                message = "Below average for this metric."
                impact = -abs(val - avg) * (1.5 if feature == 'domain_fit_score' else 1.0)

            if direction != "neutral":
                explanations.append({
                    "feature": feature,
                    "value": round(val, 1) if isinstance(val, float) else val,
                    "direction": direction,
                    "message": message,
                    "impact": impact,
                })

        explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)

        # Bug 4 Fix: if every feature equals the average, return a neutral fallback
        # so waterfall_plot never receives an empty list
        if not explanations:
            explanations.append({
                "feature": "overall_profile",
                "value": round(base_prob, 1),
                "direction": "neutral",
                "message": "Your profile closely matches the average candidate.",
                "impact": 0.0,
            })

        return explanations

    def waterfall_plot(self, features_dict, final_prob):
        exps = self.explain_text(features_dict, final_prob)

        features = [e["feature"].replace('_', ' ').title() for e in exps]
        impacts = [e["impact"] for e in exps]

        # Bug 4 Fix: handle zero-impact edge case for barh
        if all(i == 0.0 for i in impacts):
            impacts = [0.01]
            features = features[:1]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in impacts]

        ax.barh(features, impacts, color=colors)
        ax.set_xlabel('Impact on Probability')
        ax.set_title('Feature Contributions (Simulated SHAP)')
        ax.grid(axis='x', linestyle='--', alpha=0.4)

        # Bug 16 Fix: use transparent backgrounds instead of hardcoded dark colour
        # so charts render correctly in both light and dark Streamlit themes
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return fig

    def radar_chart(self, features_dict):
        labels = ['Experience', 'Projects', 'Education', 'Certs', 'Domain Fit', 'Coverage']

        candidate_vals = [
            min(features_dict['experience_years'] / 8.0, 1.0),
            min(features_dict['project_count'] / 10.0, 1.0),
            features_dict['education_score'] / 2.0,
            min(features_dict['certification_count'] / 5.0, 1.0),
            features_dict['domain_fit_score'] / 100.0,
            (100 - features_dict['gap_percentage']) / 100.0,
        ]

        ideal_vals = [0.8, 0.8, 0.5, 0.6, 0.9, 0.9]

        # Close the loop for polar chart
        candidate_vals += [candidate_vals[0]]
        ideal_vals += [ideal_vals[0]]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

        ax.plot(angles, candidate_vals, color='#667eea', linewidth=2, label='You')
        ax.fill(angles, candidate_vals, color='#667eea', alpha=0.25)
        ax.plot(angles, ideal_vals, color='#2ecc71', linewidth=1,
                linestyle='dashed', label='Ideal')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])

        # Bug 16 Fix: transparent background instead of hardcoded #0e1117
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        return fig