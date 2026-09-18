"""
Shared constants for the Siege ML Live Round Predictor.

Operator names below come straight from the handoff doc's verified
"Full operator pool" section (post-cleaning, e.g. "THERMITE" not
"GIGN-THERMITE", "RECRUIT" not "GSG9-RESERVE") and match the
ATTACK_<op> / DEFENSE_<op> feature columns used in training.

Map and site naming is NOT hardcoded here — the raw dataset uses
inconsistent Season 5 Kaggle-era spellings (e.g. "CLUB_HOUSE", not
"CLUBHOUSE") that don't match in-game names, and guessing at them
caused a real bug once already. Use src.features.FeatureEncoder's
valid_maps() / valid_sites(map_name) as the source of truth instead.
"""

ATTACK_OPERATORS = [
    "ASH", "BLACKBEARD", "BLITZ", "BUCK", "CAPITAO", "FUZE", "GLAZ",
    "HIBANA", "IQ", "JACKAL", "MONTAGNE", "RECRUIT", "SLEDGE",
    "THATCHER", "THERMITE", "TWITCH",
]

DEFENSE_OPERATORS = [
    "BANDIT", "CASTLE", "CAVEIRA", "DOC", "ECHO", "FROST", "JAGER",
    "KAPKAN", "MIRA", "MUTE", "PULSE", "RECRUIT", "ROOK", "SMOKE",
    "TACHANKA", "VALKYRIE",
]