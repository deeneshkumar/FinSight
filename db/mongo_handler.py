"""
MongoDB handler for FinSight AI
Stores:
  - User preferences (risk tolerance, watchlist)
  - Analysis history per user/symbol
"""

import os
from datetime import datetime
from pymongo import MongoClient, errors


def _get_db():
    """Returns the finsight database client. Raises on connection failure."""
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client.finsight


# ─────────────────────────────────────────────
# Save Analysis Result
# ─────────────────────────────────────────────
def save_analysis(user_id: str, symbol: str, analysis: dict) -> bool:
    """
    Persists a risk analysis result to MongoDB.
    Collection: finsight.analyses
    """
    try:
        db = _get_db()
        db.analyses.insert_one({
            "user_id": user_id,
            "symbol": symbol.upper(),
            "analysis": analysis,
            "timestamp": datetime.utcnow()
        })
        return True
    except errors.ServerSelectionTimeoutError:
        print("⚠️ MongoDB unavailable — analysis not saved")
        return False


# ─────────────────────────────────────────────
# Save / Update User Preferences
# ─────────────────────────────────────────────
def save_preferences(user_id: str, preferences: dict) -> bool:
    """
    Upserts user preferences (risk tolerance, watchlist).
    Collection: finsight.user_preferences
    """
    try:
        db = _get_db()
        db.user_preferences.update_one(
            {"user_id": user_id},
            {"$set": {
                **preferences,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        return True
    except errors.ServerSelectionTimeoutError:
        print("⚠️ MongoDB unavailable — preferences not saved")
        return False


# ─────────────────────────────────────────────
# Retrieve Analysis History
# ─────────────────────────────────────────────
def get_analysis_history(user_id: str, limit: int = 10) -> list:
    """
    Returns the last N analyses for a user.
    """
    try:
        db = _get_db()
        records = db.analyses.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(records)
    except errors.ServerSelectionTimeoutError:
        return []


# ─────────────────────────────────────────────
# Get User Preferences
# ─────────────────────────────────────────────
def get_preferences(user_id: str) -> dict:
    """
    Fetches stored preferences for a user. Returns defaults if not found.
    """
    try:
        db = _get_db()
        prefs = db.user_preferences.find_one({"user_id": user_id}, {"_id": 0})
        return prefs or {"risk_tolerance": 5, "watchlist": []}
    except errors.ServerSelectionTimeoutError:
        return {"risk_tolerance": 5, "watchlist": []}