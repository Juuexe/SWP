"""
Rebuilds a single round's feature vector exactly the way
01_explore_data.ipynb built the training data:

  - 208 base features: ATTACK_<op>, DEFENSE_<op>, MAP_<name>, MAPSITE_<MAP>__<SITE>
  - 256 matchup features: MATCHUP_<attacker>_VS_<defender>

Nothing about map/site naming is hardcoded here. The raw dataset uses
Season 5 Kaggle-era naming (e.g. "CLUB_HOUSE" not "CLUBHOUSE",
"BARTLETT_U." not "BARTLETT UNIVERSITY"), and site values often include
compound dual-site strings joined with "-". Rather than guess at that
formatting, this module reads models/feature_cols.joblib directly and
treats it as ground truth for what's valid.
"""

from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class FeatureEncoder:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.feature_cols = joblib.load(models_dir / "feature_cols.joblib")
        self.matchup_feature_names = joblib.load(
            models_dir / "matchup_feature_names.joblib"
        )
        self._base_index = {name: i for i, name in enumerate(self.feature_cols)}
        self._matchup_index = {
            name: i for i, name in enumerate(self.matchup_feature_names)
        }

        # Derive valid maps and map->sites straight from the trained
        # feature columns, so this never drifts out of sync with what
        # the model actually knows.
        self._valid_maps = sorted(
            c[len("MAP_"):]
            for c in self.feature_cols
            if c.startswith("MAP_") and not c.startswith("MAPSITE_")
        )

        self._sites_by_map: dict[str, list[str]] = {m: [] for m in self._valid_maps}
        for c in self.feature_cols:
            if c.startswith("MAPSITE_") and "__" in c:
                body = c[len("MAPSITE_"):]
                map_part, site_part = body.split("__", 1)
                if map_part in self._sites_by_map:
                    self._sites_by_map[map_part].append(site_part)
        for m in self._sites_by_map:
            self._sites_by_map[m].sort()

    def valid_maps(self) -> list[str]:
        return list(self._valid_maps)

    def valid_sites(self, map_name: str) -> list[str]:
        return list(self._sites_by_map.get(map_name, []))

    def encode(
        self,
        attackers: list[str],
        defenders: list[str],
        map_name: str,
        site: str,
    ) -> csr_matrix:
        """
        attackers / defenders: exactly 5 operator names each, already
            cleaned (e.g. "THERMITE" not "GIGN-THERMITE", "RECRUIT" not
            "GSG9-RESERVE").
        map_name: exact raw dataset spelling, e.g. "CLUB_HOUSE",
            "BARTLETT_U.", "FAVELAS". Use valid_maps() to see all 16.
        site: exact raw dataset objectivelocation value for that map,
            e.g. "VAULT" or a compound "LOCKERS-CCTV_ROOM". Use
            valid_sites(map_name) to see options for a given map.

        Returns a single-row sparse matrix with base + matchup features
        concatenated, ready to feed the matchup-enhanced model.
        """
        if len(attackers) != 5:
            raise ValueError(f"Expected 5 attackers, got {len(attackers)}")
        if len(defenders) != 5:
            raise ValueError(f"Expected 5 defenders, got {len(defenders)}")

        base_vec = np.zeros(len(self.feature_cols), dtype="int8")

        for op in attackers:
            col = f"ATTACK_{op}"
            if col not in self._base_index:
                raise ValueError(f"Unknown attacker feature: {col}")
            base_vec[self._base_index[col]] = 1

        for op in defenders:
            col = f"DEFENSE_{op}"
            if col not in self._base_index:
                raise ValueError(f"Unknown defender feature: {col}")
            base_vec[self._base_index[col]] = 1

        map_col = f"MAP_{map_name}"
        if map_col not in self._base_index:
            raise ValueError(
                f"Unknown map: {map_name!r}. Valid maps: {self._valid_maps}"
            )
        base_vec[self._base_index[map_col]] = 1

        mapsite_col = f"MAPSITE_{map_name}__{site}"
        if mapsite_col in self._base_index:
            base_vec[self._base_index[mapsite_col]] = 1
        else:
            # Not fatal: the model still has map + all 10 operators to
            # go on. But this usually means the site string didn't match
            # exactly — surface the valid options so it's easy to fix.
            print(
                f"[warning] Unrecognized site {site!r} for map {map_name!r}. "
                f"Valid sites for this map: {self._sites_by_map.get(map_name, [])}"
            )

        matchup_vec = np.zeros(len(self.matchup_feature_names), dtype="int8")
        for attacker in attackers:
            for defender in defenders:
                key = f"MATCHUP_{attacker}_VS_{defender}"
                idx = self._matchup_index.get(key)
                if idx is not None:
                    matchup_vec[idx] = 1

        return hstack(
            [csr_matrix(base_vec.reshape(1, -1)), csr_matrix(matchup_vec.reshape(1, -1))],
            format="csr",
        )