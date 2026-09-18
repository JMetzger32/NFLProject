"""Home-stadium coordinates and time zones, for travel and rest context.

The database has no stadium coordinates, so this static lookup supplies them. Travel
is approximated as the great-circle distance between the visiting team's home
stadium and the host's, which is what matters for the fatigue/jet-lag signal; it
ignores neutral-site games, which are rare and flagged separately by `location`.

UTC offsets are standard time; the small DST effect is irrelevant at the resolution
of a "how many time zones did you cross" feature.
"""

from math import asin, cos, radians, sin, sqrt

# team -> (latitude, longitude, UTC offset hours)
STADIUMS: dict[str, tuple[float, float, int]] = {
    "ARI": (33.5276, -112.2626, -7),
    "ATL": (33.7554, -84.4008, -5),
    "BAL": (39.2780, -76.6227, -5),
    "BUF": (42.7738, -78.7870, -5),
    "CAR": (35.2258, -80.8528, -5),
    "CHI": (41.8623, -87.6167, -6),
    "CIN": (39.0955, -84.5161, -5),
    "CLE": (41.5061, -81.6995, -5),
    "DAL": (32.7473, -97.0945, -6),
    "DEN": (39.7439, -105.0201, -7),
    "DET": (42.3400, -83.0456, -5),
    "GB": (44.5013, -88.0622, -6),
    "HOU": (29.6847, -95.4107, -6),
    "IND": (39.7601, -86.1639, -5),
    "JAX": (30.3240, -81.6373, -5),
    "KC": (39.0489, -94.4839, -6),
    "LA": (33.9535, -118.3392, -8),
    "LAC": (33.9535, -118.3392, -8),
    "LV": (36.0909, -115.1833, -8),
    "MIA": (25.9580, -80.2389, -5),
    "MIN": (44.9735, -93.2575, -6),
    "NE": (42.0909, -71.2643, -5),
    "NO": (29.9511, -90.0812, -6),
    "NYG": (40.8135, -74.0745, -5),
    "NYJ": (40.8135, -74.0745, -5),
    "PHI": (39.9008, -75.1675, -5),
    "PIT": (40.4468, -80.0158, -5),
    "SEA": (47.5952, -122.3316, -8),
    "SF": (37.4030, -121.9700, -8),
    "TB": (27.9759, -82.5033, -5),
    "TEN": (36.1665, -86.7713, -6),
    "WAS": (38.9077, -76.8645, -5),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(a))


def travel_miles(team: str, host: str) -> float:
    """Distance the team travelled to reach the host's stadium. 0 when at home."""
    if team == host:
        return 0.0
    if team not in STADIUMS or host not in STADIUMS:
        return float("nan")
    lat1, lon1, _ = STADIUMS[team]
    lat2, lon2, _ = STADIUMS[host]
    return haversine_miles(lat1, lon1, lat2, lon2)


def timezone_shift(team: str, host: str) -> float:
    """Time zones crossed. Negative = travelling east (the harder direction)."""
    if team not in STADIUMS or host not in STADIUMS:
        return float("nan")
    return float(STADIUMS[host][2] - STADIUMS[team][2])
