#!/usr/bin/env python3
"""
Shared helpers for V4 extraction scripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Path to the line-delimited JSON source file
SOURCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "source"
    / "last_full_data_decidento.json"
)


def iter_source() -> Iterable[Dict]:
    """Yield each `computed` dict from the line-delimited source file."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}\n"
            f"Please ensure the source file exists at: {SOURCE_FILE}"
        )
    
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            computed = obj.get("computed", {})
            if computed:
                yield computed


def load_source() -> List[Dict]:
    """Load all computed records into memory."""
    return list(iter_source())


def to_timestamp(iso_str: Optional[str]) -> Optional[int]:
    """Convert ISO datetime strings to Unix timestamps."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def normalize_dept(dept: Optional[str]) -> str:
    """Ensure department codes keep a leading zero when needed."""
    if dept is None:
        return ""
    dept_str = str(dept)
    if len(dept_str) == 1:
        return f"0{dept_str}"
    return dept_str
