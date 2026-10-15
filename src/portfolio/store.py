"""SQLite storage for investor profiles, plus two synthetic demo profiles."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.contracts import Holding, Profile

DEFAULT_DB = "data/portfolio.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    risk_preference TEXT NOT NULL,
    horizon_months INTEGER NOT NULL,
    max_single_position_pct REAL NOT NULL,
    max_sector_pct REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS holdings (
    profile_id TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    pos INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL,
    sector TEXT NOT NULL,
    PRIMARY KEY (profile_id, pos)
);
"""


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_profile(conn: sqlite3.Connection, p: Profile) -> None:
    with conn:
        conn.execute("DELETE FROM holdings WHERE profile_id = ?", (p.profile_id,))
        conn.execute("DELETE FROM profiles WHERE profile_id = ?", (p.profile_id,))
        conn.execute("INSERT INTO profiles VALUES (?, ?, ?, ?, ?)",
                     (p.profile_id, p.risk_preference, p.horizon_months,
                      p.max_single_position_pct, p.max_sector_pct))
        conn.executemany(
            "INSERT INTO holdings VALUES (?, ?, ?, ?, ?, ?)",
            [(p.profile_id, i, h.symbol, h.quantity, h.avg_cost, h.sector)
             for i, h in enumerate(p.holdings)])


def load_profile(conn: sqlite3.Connection, profile_id: str) -> Profile | None:
    row = conn.execute(
        "SELECT risk_preference, horizon_months, max_single_position_pct, max_sector_pct "
        "FROM profiles WHERE profile_id = ?", (profile_id,)).fetchone()
    if row is None:
        return None
    rows = conn.execute(
        "SELECT symbol, quantity, avg_cost, sector FROM holdings "
        "WHERE profile_id = ? ORDER BY pos", (profile_id,)).fetchall()
    return Profile(profile_id=profile_id, risk_preference=row[0], horizon_months=row[1],
                   max_single_position_pct=row[2], max_sector_pct=row[3],
                   holdings=[Holding(symbol=s, quantity=q, avg_cost=c, sector=sec)
                             for s, q, c, sec in rows])


def list_profile_ids(conn: sqlite3.Connection) -> list:
    return [r[0] for r in conn.execute("SELECT profile_id FROM profiles ORDER BY profile_id")]


# SYNTHETIC demo data: made-up quantities and costs, not real investors.
DEMO_PROFILES = [
    Profile(profile_id="conservative", risk_preference="low", horizon_months=60,
            max_single_position_pct=20.0, max_sector_pct=40.0, holdings=[
                Holding(symbol="HDFCBANK", quantity=100, avg_cost=1600, sector="Banking"),
                Holding(symbol="ICICIBANK", quantity=100, avg_cost=1100, sector="Banking"),
                Holding(symbol="ITC", quantity=500, avg_cost=400, sector="FMCG"),
                Holding(symbol="TCS", quantity=50, avg_cost=3500, sector="IT"),
            ]),
    Profile(profile_id="aggressive", risk_preference="high", horizon_months=24,
            max_single_position_pct=35.0, max_sector_pct=60.0, holdings=[
                Holding(symbol="TCS", quantity=200, avg_cost=3400, sector="IT"),
                Holding(symbol="INFY", quantity=150, avg_cost=1500, sector="IT"),
                Holding(symbol="RELIANCE", quantity=100, avg_cost=2500, sector="Energy"),
                Holding(symbol="SBIN", quantity=300, avg_cost=700, sector="Banking"),
            ]),
]


def seed_demo_profiles(conn: sqlite3.Connection) -> None:
    for p in DEMO_PROFILES:
        save_profile(conn, p)


if __name__ == "__main__":
    c = connect()
    seed_demo_profiles(c)
    print("profiles in", DEFAULT_DB, ":", list_profile_ids(c))
