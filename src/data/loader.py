"""JSON data loading utilities"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_json_file(path: Path) -> List[Dict]:
    """Load a single JSON file and return as list of dictionaries."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
    except Exception as e:
        print(f"⚠️  Error loading {path}: {e}")
        return []


def load_all_json_data(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all JSON data files from data directory.
    
    Args:
        data_dir: Path to directory containing JSON files
        
    Returns:
        Dictionary mapping dataset names to lists of records
    """
    files = {
        "company_basic": "01_company_basic_info.json",
        "financial": "02_financial_data.json",
        "workforce": "03_workforce_data.json",
        "structure": "04_company_structure.json",
        "flags": "05_classification_flags.json",
        "contact": "06_contact_metrics.json",
        "kpi": "07_kpi_data.json",
        "signals": "08_signals.json",
        "articles": "09_articles.json",
    }
    
    out = {}
    for key, filename in files.items():
        filepath = data_dir / filename
        data = load_json_file(filepath)
        out[key] = data
        if data:
            print(f"  ✓ Loaded {key}: {len(data)} records")
        else:
            print(f"  ⚠️  {key}: File not found or empty")
    
    return out


def merge_on_siren(data_dict: Dict[str, List[Dict]], preserve_panel: bool = True) -> pd.DataFrame:
    """
    Merge all datasets on SIREN identifier.
    
    For KPI data with year column, preserves panel structure (multiple rows per company)
    if preserve_panel=True, otherwise aggregates to latest year.
    
    Args:
        data_dict: Dictionary of dataset names to lists of records
        preserve_panel: If True and KPI has year column, preserve panel structure.
                       If False, aggregate to latest year per company.
        
    Returns:
        Merged DataFrame with all company features
    """
    if not data_dict.get("company_basic"):
        raise ValueError("company_basic data is required")
    
    df = pd.DataFrame(data_dict["company_basic"])
    
    if "siren" in df.columns:
        df["siren"] = df["siren"].astype(str)
    
    kpi_has_year = False
    if data_dict.get("kpi"):
        kpi_df = pd.DataFrame(data_dict["kpi"])
        kpi_has_year = "year" in kpi_df.columns
    
    if kpi_has_year and preserve_panel:
        kpi_df = pd.DataFrame(data_dict["kpi"])
        if "siren" in kpi_df.columns:
            kpi_df["siren"] = kpi_df["siren"].astype(str)
        
        df = df.merge(kpi_df, on="siren", how="left", suffixes=("", "_kpi"))
        if "siren_kpi" in df.columns:
            df = df.drop(columns=["siren_kpi"])
        
        merge_keys = ["financial", "workforce", "structure", "flags", "contact"]
        for key in merge_keys:
            if data_dict.get(key):
                temp_df = pd.DataFrame(data_dict[key])
                if "siren" in temp_df.columns:
                    temp_df["siren"] = temp_df["siren"].astype(str)
                    df = df.merge(temp_df, on="siren", how="left", suffixes=("", f"_{key}"))
    else:
        merge_keys = ["financial", "workforce", "structure", "flags", "contact", "kpi"]
        for key in merge_keys:
            if data_dict.get(key):
                temp_df = pd.DataFrame(data_dict[key])
                if "siren" in temp_df.columns:
                    temp_df["siren"] = temp_df["siren"].astype(str)
                    
                    if key == "kpi" and "year" in temp_df.columns:
                        temp_df = temp_df.sort_values(["siren", "year"], ascending=[True, False])
                        temp_df = temp_df.drop_duplicates(subset="siren", keep="first")
                    
                    df = df.merge(temp_df, on="siren", how="left", suffixes=("", f"_{key}"))
    
    if data_dict.get("signals"):
        signals_df = pd.DataFrame(data_dict["signals"])
        if "siren" in signals_df.columns:
            signals_df["siren"] = signals_df["siren"].astype(str)
            signals_grouped = signals_df.groupby("siren").apply(
                lambda x: x.to_dict("records")
            ).to_dict()
            df["signals"] = df["siren"].map(signals_grouped).fillna("").apply(
                lambda x: x if isinstance(x, list) else []
            )
    else:
        df["signals"] = df["siren"].apply(lambda x: [])
    
    if data_dict.get("articles"):
        articles_df = pd.DataFrame(data_dict["articles"])
        if "siren" in articles_df.columns:
            articles_df["siren"] = articles_df["siren"].astype(str)
            articles_grouped = articles_df.groupby("siren").apply(
                lambda x: x.to_dict("records")
            ).to_dict()
            df["articles"] = df["siren"].map(articles_grouped).fillna("").apply(
                lambda x: x if isinstance(x, list) else []
            )
    else:
        df["articles"] = df["siren"].apply(lambda x: [])
    
    return df
