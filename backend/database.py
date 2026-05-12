"""
MongoDB Database Layer for Construction Site Safety Monitoring
===============================================================
Stores violation records, processed video metadata, and analytics data.
Uses pymongo to connect to a local MongoDB instance.
"""

import logging
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from bson import ObjectId

logger = logging.getLogger("database")

# ─── MongoDB Connection ───────────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "construction_safety"

_client = None
_db = None


def get_db():
    """Get or create the MongoDB connection and return the database object."""
    global _client, _db
    if _db is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            # Test the connection
            _client.server_info()
            _db = _client[DB_NAME]
            logger.info(f"Connected to MongoDB: {MONGO_URI}/{DB_NAME}")

            # Create indexes for performance
            _db.violations.create_index([("timestamp", DESCENDING)])
            _db.violations.create_index([("video_name", 1)])
            _db.violations.create_index([("violation_type", 1)])
            _db.processed_videos.create_index([("date_processed", DESCENDING)])

        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using fallback in-memory store.")
            return None
    return _db


def _serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


# ─── Violations Collection ────────────────────────────────────────────────────

def insert_violation(video_name, violation_type, risk_level, person_id=None,
                     bbox=None, details=None, frame_number=None):
    """
    Insert a new violation record into the database.

    Args:
        video_name: Name of the source video
        violation_type: e.g. "NO HELMET", "FALL DETECTED", "NO VEST"
        risk_level: "danger", "warning", or "safe"
        person_id: Tracked person identifier
        bbox: Bounding box [x1, y1, x2, y2]
        details: Additional detail string
        frame_number: Video frame number
    """
    db = get_db()
    doc = {
        "video_name": video_name,
        "violation_type": violation_type,
        "risk_level": risk_level,
        "person_id": person_id,
        "bbox": bbox,
        "details": details,
        "frame_number": frame_number,
        "timestamp": datetime.now(timezone.utc),
    }
    if db is not None:
        result = db.violations.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
    else:
        doc["_id"] = str(ObjectId())
        _fallback_violations.append(doc)
    return doc


def get_violations(video_name=None, violation_type=None, limit=200, skip=0):
    """
    Retrieve violation records, optionally filtered by video name or type.
    Returns newest first.
    """
    db = get_db()
    if db is not None:
        query = {}
        if video_name:
            query["video_name"] = video_name
        if violation_type:
            query["violation_type"] = violation_type
        cursor = db.violations.find(query).sort("timestamp", DESCENDING).skip(skip).limit(limit)
        return [_serialize_doc(doc) for doc in cursor]
    else:
        results = list(_fallback_violations)
        if video_name:
            results = [v for v in results if v.get("video_name") == video_name]
        if violation_type:
            results = [v for v in results if v.get("violation_type") == violation_type]
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[skip:skip + limit]


def get_violation_count(video_name=None):
    """Get total violation count, optionally filtered by video."""
    db = get_db()
    if db is not None:
        query = {}
        if video_name:
            query["video_name"] = video_name
        return db.violations.count_documents(query)
    else:
        if video_name:
            return sum(1 for v in _fallback_violations if v.get("video_name") == video_name)
        return len(_fallback_violations)


def clear_violations(video_name=None):
    """Clear violations, optionally only for a specific video."""
    db = get_db()
    if db is not None:
        query = {}
        if video_name:
            query["video_name"] = video_name
        db.violations.delete_many(query)
    else:
        if video_name:
            _fallback_violations[:] = [
                v for v in _fallback_violations if v.get("video_name") != video_name
            ]
        else:
            _fallback_violations.clear()


# ─── Processed Videos Collection ──────────────────────────────────────────────

def insert_processed_video(video_name, total_frames, total_violations, risk_level,
                           duration_seconds=0):
    """Record metadata about a processed video."""
    db = get_db()
    doc = {
        "video_name": video_name,
        "date_processed": datetime.now(timezone.utc),
        "total_frames": total_frames,
        "total_violations": total_violations,
        "risk_level": risk_level,
        "duration_seconds": duration_seconds,
    }
    if db is not None:
        result = db.processed_videos.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
    else:
        doc["_id"] = str(ObjectId())
        _fallback_videos.append(doc)
    return doc


def get_processed_videos(limit=50):
    """Get list of processed videos, newest first."""
    db = get_db()
    if db is not None:
        cursor = db.processed_videos.find().sort("date_processed", DESCENDING).limit(limit)
        return [_serialize_doc(doc) for doc in cursor]
    else:
        results = sorted(_fallback_videos, key=lambda x: x.get("date_processed", ""), reverse=True)
        return results[:limit]


# ─── Analytics Helpers ─────────────────────────────────────────────────────────

def get_analytics_summary(video_name=None):
    """
    Get aggregated analytics data for the analytics page.
    Returns violation counts by type, risk distribution, etc.
    """
    db = get_db()
    if db is not None:
        match_stage = {}
        if video_name:
            match_stage = {"video_name": video_name}

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {
                "_id": "$violation_type",
                "count": {"$sum": 1},
            }},
            {"$sort": {"count": -1}},
        ]
        type_counts = list(db.violations.aggregate(pipeline))

        risk_pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {
                "_id": "$risk_level",
                "count": {"$sum": 1},
            }},
        ]
        risk_counts = list(db.violations.aggregate(risk_pipeline))

        total = get_violation_count(video_name)

        return {
            "total_violations": total,
            "by_type": {item["_id"]: item["count"] for item in type_counts},
            "by_risk": {item["_id"]: item["count"] for item in risk_counts},
            "processed_videos": len(get_processed_videos()),
        }
    else:
        # Fallback in-memory analytics
        violations = _fallback_violations
        if video_name:
            violations = [v for v in violations if v.get("video_name") == video_name]

        by_type = {}
        by_risk = {}
        for v in violations:
            vt = v.get("violation_type", "unknown")
            rl = v.get("risk_level", "unknown")
            by_type[vt] = by_type.get(vt, 0) + 1
            by_risk[rl] = by_risk.get(rl, 0) + 1

        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_risk": by_risk,
            "processed_videos": len(_fallback_videos),
        }


# ─── In-memory fallback (used when MongoDB is unavailable) ────────────────────
_fallback_violations = []
_fallback_videos = []
