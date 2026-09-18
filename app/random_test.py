"""
Random round generator for exercising the prediction pipeline end to end.

Picks 5 random attackers, 5 random defenders, a random map, and a
random site valid for that map (all sourced live from the trained
feature columns), runs the prediction, and repeats for N iterations.

Run from the SWP/ project root:
    python app/random_test.py            # 10 iterations by default
    python app/random_test.py --n 25      # custom iteration count
    python app/random_test.py --seed 42   # reproducible picks
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ATTACK_OPERATORS, DEFENSE_OPERATORS  # noqa: E402
from src.predict import RoundPredictor  # noqa: E402


def random_round(predictor: RoundPredictor, rng: random.Random) -> dict:
    attackers = rng.sample(ATTACK_OPERATORS, 5)
    defenders = rng.sample(DEFENSE_OPERATORS, 5)

    valid_maps = predictor.encoder.valid_maps()
    map_name = rng.choice(valid_maps)

    valid_sites = predictor.encoder.valid_sites(map_name)
    site = rng.choice(valid_sites) if valid_sites else ""

    result = predictor.predict(attackers, defenders, map_name, site)

    return {
        "attackers": attackers,
        "defenders": defenders,
        "map": map_name,
        "site": site,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description="Random round prediction test")
    parser.add_argument("--n", type=int, default=10, help="number of iterations")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    predictor = RoundPredictor()

    print(f"Running {args.n} random rounds through the matchup-enhanced model\n")

    for i in range(1, args.n + 1):
        r = random_round(predictor, rng)
        print(f"--- Round {i} ---")
        print(f"Attackers: {', '.join(r['attackers'])}")
        print(f"Defenders: {', '.join(r['defenders'])}")
        site_display = f"{r['map']} / {r['site']}" if r["site"] else r["map"]
        print(f"Map / Site: {site_display}")
        print(f"Attack win probability:  {r['attack_win_probability'] * 100:.1f}%")
        print(f"Defense win probability: {r['defense_win_probability'] * 100:.1f}%")
        print()


if __name__ == "__main__":
    main()
