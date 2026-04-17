"""
Premier League Historical Data Downloader & Converter
======================================================
Downloads PL data from football-data.co.uk (free, no API key needed)
and converts it to match your existing training CSV schema:

    match_id, season, gameweek, match_date, home_team, away_team,
    home_score, away_score, home_goals, away_goals, status, result

Usage:
    python download_pl_historical.py

Output files (in ./output/):
    pl_2010_historical.csv   (2010-11 season)
    pl_2011_historical.csv   (2011-12 season)
    ...
    pl_2022_historical.csv   (2022-23 season)

These match the exact format of your existing pl_2023_historical.csv
and pl_2024_historical.csv files, so you can drop them straight into
your RAW_DIR and retrain with 10+ seasons of data.
"""

import requests
import pandas as pd
import time
import re
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Seasons to download ───────────────────────────────────────────────────────
# Format: (start_year, "YYYY-YY label", "output filename season label")
# football-data.co.uk URL format: /mmz4281/YYZZ/E0.csv
# e.g. 2010-11 → /mmz4281/1011/E0.csv
SEASONS = [
    (2010, "10", "11", "2010-11"),
    (2011, "11", "12", "2011-12"),
    (2012, "12", "13", "2012-13"),
    (2013, "13", "14", "2013-14"),
    (2014, "14", "15", "2014-15"),
    (2015, "15", "16", "2015-16"),
    (2016, "16", "17", "2016-17"),
    (2017, "17", "18", "2017-18"),
    (2018, "18", "19", "2018-19"),
    (2019, "19", "20", "2019-20"),
    (2020, "20", "21", "2020-21"),
    (2021, "21", "22", "2021-22"),
    (2022, "22", "23", "2022-23"),
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{yy1}{yy2}/E0.csv"

# ── Team name normalisation ───────────────────────────────────────────────────
# football-data.co.uk uses short names; we normalise to the full FC names
# used in your existing dataset
TEAM_MAP = {
    "Arsenal":          "Arsenal FC",
    "Aston Villa":      "Aston Villa FC",
    "Bournemouth":      "AFC Bournemouth",
    "Brentford":        "Brentford FC",
    "Brighton":         "Brighton & Hove Albion FC",
    "Burnley":          "Burnley FC",
    "Chelsea":          "Chelsea FC",
    "Crystal Palace":   "Crystal Palace FC",
    "Everton":          "Everton FC",
    "Fulham":           "Fulham FC",
    "Huddersfield":     "Huddersfield Town AFC",
    "Hull":             "Hull City AFC",
    "Leeds":            "Leeds United FC",
    "Leicester":        "Leicester City FC",
    "Liverpool":        "Liverpool FC",
    "Luton":            "Luton Town FC",
    "Man City":         "Manchester City FC",
    "Man United":       "Manchester United FC",
    "Middlesbrough":    "Middlesbrough FC",
    "Newcastle":        "Newcastle United FC",
    "Norwich":          "Norwich City FC",
    "Nott'm Forest":    "Nottingham Forest FC",
    "QPR":              "Queens Park Rangers FC",
    "Reading":          "Reading FC",
    "Sheffield United": "Sheffield United FC",
    "Southampton":      "Southampton FC",
    "Stoke":            "Stoke City FC",
    "Sunderland":       "Sunderland AFC",
    "Swansea":          "Swansea City AFC",
    "Tottenham":        "Tottenham Hotspur FC",
    "Watford":          "Watford FC",
    "West Brom":        "West Bromwich Albion FC",
    "West Ham":         "West Ham United FC",
    "Wigan":            "Wigan Athletic FC",
    "Wolves":           "Wolverhampton Wanderers FC",
    # edge-case variants
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Manchester City":        "Manchester City FC",
    "Manchester United":      "Manchester United FC",
    "Newcastle United":       "Newcastle United FC",
    "Nottingham Forest":      "Nottingham Forest FC",
    "Sheffield Weds":         "Sheffield Wednesday FC",
    "Blackburn":              "Blackburn Rovers FC",
    "Blackpool":              "Blackpool FC",
    "Bolton":                 "Bolton Wanderers FC",
    "Cardiff":                "Cardiff City FC",
    "Charlton":               "Charlton Athletic FC",
    "Coventry":               "Coventry City FC",
    "Derby":                  "Derby County FC",
    "Ipswich":                "Ipswich Town FC",
    "Sunderland AFC":         "Sunderland AFC",
    "Wolvs":                  "Wolverhampton Wanderers FC",
    "Leeds United":           "Leeds United FC",
}

def normalise_team(name: str) -> str:
    """Return full team name; fall back to original if not in map."""
    return TEAM_MAP.get(name.strip(), name.strip())


def download_season(yy1: str, yy2: str, season_label: str) -> pd.DataFrame | None:
    """Download one season CSV from football-data.co.uk and return raw DataFrame."""
    url = BASE_URL.format(yy1=yy1, yy2=yy2)
    print(f"  Downloading {season_label} → {url}")
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # Decode — file is latin-1 / windows-1252
        from io import StringIO
        df = pd.read_csv(StringIO(resp.content.decode("latin-1")))
        print(f"    ✅ {len(df)} rows, columns: {list(df.columns[:10])} ...")
        return df
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return None


def convert_to_schema(df: pd.DataFrame, season_label: str, start_year: int) -> pd.DataFrame:
    """
    Convert football-data.co.uk format to your training schema.

    football-data columns we use:
        Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
        (also HTHG, HTAG if present)

    Output columns (matches your existing CSVs):
        match_id, season, gameweek, match_date,
        home_team, away_team, home_score, away_score,
        home_goals, away_goals, status, result
    """
    df = df.copy()

    # ── Drop rows with missing critical columns ────────────────────────────
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if df.empty:
        return pd.DataFrame()

    # ── Parse date ────────────────────────────────────────────────────────
    # football-data uses DD/MM/YY or DD/MM/YYYY
    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return pd.to_datetime(d, format=fmt)
            except Exception:
                pass
        return pd.NaT

    df["match_date_parsed"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["match_date_parsed"])

    # ── Sort chronologically — use as proxy for gameweek ─────────────────
    df = df.sort_values("match_date_parsed").reset_index(drop=True)

    # Estimate gameweek: every ~10 matches on the same date cluster = 1 GW
    # Simple approach: assign GW based on date rank groups (10 per GW)
    unique_dates = sorted(df["match_date_parsed"].unique())
    date_to_gw = {}
    gw = 1
    batch = []
    for d in unique_dates:
        batch.append(d)
        total = sum(
            len(df[df["match_date_parsed"] == b]) for b in batch
        )
        date_to_gw[d] = gw
        if total >= 9:           # ~10 matches = 1 gameweek round
            gw += 1
            batch = []
    if batch:                    # remainder → same GW
        for d in batch:
            date_to_gw[d] = gw

    df["gameweek"] = df["match_date_parsed"].map(date_to_gw)

    # ── Map result ────────────────────────────────────────────────────────
    result_map = {"H": "H", "D": "D", "A": "A"}
    df["result"] = df["FTR"].map(result_map)

    # ── Build output ──────────────────────────────────────────────────────
    out = pd.DataFrame()
    out["match_id"] = (
        str(start_year) + df.index.astype(str).str.zfill(4)
    ).astype(int)
    out["season"]     = season_label
    out["gameweek"]   = df["gameweek"]
    out["match_date"] = df["match_date_parsed"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["home_team"]  = df["HomeTeam"].apply(normalise_team)
    out["away_team"]  = df["AwayTeam"].apply(normalise_team)
    out["home_score"] = df["FTHG"].astype(int)
    out["away_score"] = df["FTAG"].astype(int)
    out["home_goals"] = out["home_score"]
    out["away_goals"] = out["away_score"]
    out["status"]     = "FINISHED"
    out["result"]     = df["result"]

    return out


def main():
    print("="*70)
    print("PL HISTORICAL DATA DOWNLOADER")
    print("Source: football-data.co.uk  |  Output: ./output/")
    print("="*70)

    all_frames = []
    success = 0

    for (start_year, yy1, yy2, season_label) in SEASONS:
        print(f"\n📅 Season {season_label}")

        raw = download_season(yy1, yy2, season_label)
        if raw is None:
            continue

        converted = convert_to_schema(raw, season_label, start_year)
        if converted.empty:
            print(f"    ⚠️  No usable rows after conversion.")
            continue

        # Save individual season file
        fname = f"pl_{start_year}_historical.csv"
        fpath = OUTPUT_DIR / fname
        converted.to_csv(fpath, index=False)
        print(f"    💾 Saved {fname}  ({len(converted)} matches)")

        all_frames.append(converted)
        success += 1

        # Be polite — don't hammer the server
        time.sleep(1.2)

    # ── Save a combined mega-file for convenience ──────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUTPUT_DIR / "pl_2010_2022_all_historical.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*70}")
        print(f"✅ Done!  {success}/{len(SEASONS)} seasons downloaded.")
        print(f"📦 Combined file: {combined_path}  ({len(combined)} total matches)")
        print(f"📁 Individual files in: {OUTPUT_DIR.resolve()}")
        print("="*70)
        print("\n📋 NEXT STEPS:")
        print("  1. Copy all pl_YYYY_historical.csv files to your RAW_DIR:")
        print("     bhakundo-predictor/required/data/raw/pl/")
        print("  2. Add them to load_all_historical_data() in train.py:")
        print("     RAW_DIR / 'pl_2010_historical.csv',")
        print("     RAW_DIR / 'pl_2011_historical.csv',")
        print("     ... (2010 through 2022)")
        print("  3. Re-run training — you'll have ~3,000+ matches to train on.")
    else:
        print("\n❌ No data downloaded. Check your internet connection.")


if __name__ == "__main__":
    main()