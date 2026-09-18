"""
Prediction entry point for the historical Season 5 model.

Usage:
    from src.predict import RoundPredictor

    predictor = RoundPredictor()
    result = predictor.predict(
        attackers=["THERMITE", "HIBANA", "BUCK", "ASH", "TWITCH"],
        defenders=["BANDIT", "MUTE", "JAGER", "MIRA", "SMOKE"],
        map_name="CLUBHOUSE",
        site="CCTV_CASH",
    )
    print(result)
    # {"attack_win_probability": 0.624, "defense_win_probability": 0.376}

This wraps the matchup-enhanced Logistic Regression (models/matchup_logreg.joblib),
the best-performing model so far (54.8% accuracy, calibration_curve-checked).
Swap in a different model file here later without touching features.py or
the app layer, as long as its input feature order matches feature_cols +
matchup_feature_names.
"""

from pathlib import Path
import joblib

from src.features import FeatureEncoder

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class RoundPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR, model_file: str = "matchup_logreg.joblib"):
        self.model = joblib.load(models_dir / model_file)
        self.encoder = FeatureEncoder(models_dir=models_dir)

    def predict(
        self,
        attackers: list[str],
        defenders: list[str],
        map_name: str,
        site: str,
    ) -> dict:
        X = self.encoder.encode(attackers, defenders, map_name, site)
        attack_prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "attack_win_probability": round(attack_prob, 4),
            "defense_win_probability": round(1 - attack_prob, 4),
        }
