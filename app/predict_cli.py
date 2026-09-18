"""
Manual-entry CLI for the Siege ML Live Round Predictor (historical model).

Map and site choices are validated against the real feature columns
(via FeatureEncoder), not a guessed list — so this can't drift out of
sync with what the model actually knows again.

Run from the SWP/ project root:
    python app/predict_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ATTACK_OPERATORS, DEFENSE_OPERATORS  # noqa: E402
from src.predict import RoundPredictor  # noqa: E402


def prompt_operators(label: str, valid_ops: list[str]) -> list[str]:
    print(f"\n{label} — choose exactly 5, comma-separated.")
    print(f"Valid options: {', '.join(valid_ops)}")
    while True:
        raw = input("> ").strip().upper()
        picks = [p.strip() for p in raw.split(",") if p.strip()]
        if len(picks) != 5:
            print(f"Need exactly 5, got {len(picks)}. Try again.")
            continue
        invalid = [p for p in picks if p not in valid_ops]
        if invalid:
            print(f"Not recognized: {invalid}. Try again.")
            continue
        return picks


def prompt_map(valid_maps: list[str]) -> str:
    print(f"\nMap — choose one.")
    print(f"Valid options: {', '.join(valid_maps)}")
    while True:
        raw = input("> ").strip().upper()
        if raw in valid_maps:
            return raw
        print(f"Not recognized: {raw!r}. Try again.")


def prompt_site(map_name: str, valid_sites: list[str]) -> str:
    print(f"\nSite for {map_name} — choose one.")
    print(f"Valid options: {', '.join(valid_sites)}")
    while True:
        raw = input("> ").strip().upper()
        if raw in valid_sites:
            return raw
        print(f"Not recognized: {raw!r}. Try again.")


def main():
    print("Siege ML Live Round Predictor (historical Season 5 model)")

    predictor = RoundPredictor()

    attackers = prompt_operators("Attackers", ATTACK_OPERATORS)
    defenders = prompt_operators("Defenders", DEFENSE_OPERATORS)

    valid_maps = predictor.encoder.valid_maps()
    map_name = prompt_map(valid_maps)

    valid_sites = predictor.encoder.valid_sites(map_name)
    if valid_sites:
        site = prompt_site(map_name, valid_sites)
    else:
        print(f"\nNo known site combos for {map_name} in training data — skipping site.")
        site = ""

    result = predictor.predict(attackers, defenders, map_name, site)

    print("\n--- Prediction ---")
    print(f"Attack win probability:  {result['attack_win_probability'] * 100:.1f}%")
    print(f"Defense win probability: {result['defense_win_probability'] * 100:.1f}%")


if __name__ == "__main__":
    main()