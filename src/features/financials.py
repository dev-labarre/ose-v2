"""
Financial ratios processing.
Loads INPI ratios, normalizes, winsorizes, and generates 3-year summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json
from datetime import datetime, timezone

from src.features import signals as signal_utils

T0 = pd.Timestamp('2023-01-01')  # timezone-naive


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a series at specified percentiles.
    
    Args:
        series: Series to winsorize
        lower: Lower percentile (default 0.01)
        upper: Upper percentile (default 0.99)
        
    Returns:
        Winsorized series
    """
    if series.isna().all():
        return series
    
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    
    return series.clip(lower=lower_bound, upper=upper_bound)


def normalize_by_size(series: pd.Series, size_series: pd.Series) -> pd.Series:
    """
    Normalize a series by company size (revenue).
    
    Args:
        series: Series to normalize
        size_series: Size series (e.g., revenue)
        
    Returns:
        Normalized series
    """
    # Avoid division by zero
    size_series = size_series.replace(0, np.nan)
    normalized = series / size_series
    return normalized


def compute_3y_summaries(df_ratios: pd.DataFrame, siren: str, 
                         date_col: str = 'date_cloture_exercice',
                         value_col: str = 'ca') -> Dict:
    """
    Compute 3-year rolling summaries for a company.
    
    Args:
        df_ratios: DataFrame with ratios
        siren: SIREN identifier
        date_col: Date column name
        value_col: Value column name
        
    Returns:
        Dictionary with last, mean, std, slope
    """
    company_ratios = df_ratios[df_ratios['siren'] == siren].copy()
    
    if len(company_ratios) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    # Sort by date
    company_ratios = company_ratios.sort_values(date_col)
    
    # Get last 3 years (before t0)
    company_ratios = company_ratios[company_ratios[date_col] < T0]
    
    if len(company_ratios) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    # Get last 3 years
    last_3y = company_ratios.tail(3)
    
    if len(last_3y) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    values = last_3y[value_col].dropna()
    
    if len(values) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    last = values.iloc[-1] if len(values) > 0 else np.nan
    mean = values.mean()
    std = values.std() if len(values) > 1 else 0.0
    
    # Compute slope (linear trend)
    if len(values) >= 2:
        x = np.arange(len(values))
        slope = np.polyfit(x, values.values, 1)[0] if len(values) >= 2 else 0.0
    else:
        slope = 0.0
    
    return {
        'last': float(last) if pd.notna(last) else np.nan,
        'mean': float(mean) if pd.notna(mean) else np.nan,
        'std': float(std) if pd.notna(std) else np.nan,
        'slope': float(slope) if pd.notna(slope) else np.nan
    }


def _first_existing(df: pd.DataFrame, candidates):
    """Return the first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_year(series: pd.Series) -> pd.Series:
    """Parse a column to integer year if possible."""
    return pd.to_numeric(series, errors='coerce').astype('Int64')


def compute_decidento_score(df: pd.DataFrame, default_sector_coeff: float = 1.3) -> pd.DataFrame:
    """
    Business-weighted signal score (adapted from coworker notebook).
    Adds: signal_score, age_bonus, etab_bonus, decidento_base_score, decidento_score.
    """
    df = df.copy()

    signal_weights = {
        "signal_B_count": 25,
        "signal_W_count": 15,
        "signal_E_count": 20,
        "signal_F_count": 15,
        "signal_N_count": 10,
        "signal_S_count": 7,
        "signal_K1_count": 8,
        "signal_I_count": -15,
        "signal_M_count": -20,
        "signal_O_count": -50,
    }

    for col in signal_weights:
        if col not in df:
            df[col] = 0

    signal_score = pd.Series(0, index=df.index)
    for col, weight in signal_weights.items():
        presence = (df[col].fillna(0) > 0).astype(int)
        signal_score += weight * presence
    df["signal_score"] = signal_score

    age = df["age_entreprise"].fillna(-1) if "age_entreprise" in df else pd.Series(-1, index=df.index)
    df["age_bonus"] = 0
    df.loc[age.between(4, 10), "age_bonus"] = 2
    df.loc[age >= 20, "age_bonus"] = 5

    if "nbEtabSecondaire" not in df:
        df["nbEtabSecondaire"] = 0
    df["etab_bonus"] = (df["nbEtabSecondaire"].fillna(0) > 0).astype(int) * 5

    if "sector_coeff" not in df:
        df["sector_coeff"] = default_sector_coeff
    df["sector_coeff"] = df["sector_coeff"].fillna(default_sector_coeff)

    df["decidento_base_score"] = df["signal_score"] + df["age_bonus"] + df["etab_bonus"]
    df["decidento_score"] = df["decidento_base_score"] * df["sector_coeff"]
    return df


def compute_financial_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Financial score (growth + profitability) adapted from coworker notebook.
    Expects columns: siren, year, ca_final, resultat_final, effectif, capital_social.
    """
    df = df.sort_values(["siren", "year"]).copy()
    eps = 1e-6

    df["ca_growth"] = df.groupby("siren")["ca_final"].pct_change().clip(-1, 1)
    df["effectif_growth"] = df.groupby("siren")["effectif"].pct_change().clip(-1, 1)
    df[["ca_growth", "effectif_growth"]] = df[["ca_growth", "effectif_growth"]].fillna(0)
    df["growth_score"] = 0.7 * df["ca_growth"] + 0.3 * df["effectif_growth"]

    df["margin_norm"] = (df["resultat_final"] / (df["ca_final"] + eps)).clip(-1, 1)
    df["capital_ratio"] = (df["resultat_final"] / (df["capital_social"] + eps)).clip(-1, 1)
    df[["margin_norm", "capital_ratio"]] = df[["margin_norm", "capital_ratio"]].fillna(0)
    df["profit_score"] = 0.7 * df["margin_norm"] + 0.3 * df["capital_ratio"]

    df["financial_score"] = 0.6 * df["growth_score"] + 0.4 * df["profit_score"]
    return df


def compute_total_ose_score(df: pd.DataFrame, w_decidento: float = 0.4, w_financial: float = 0.6) -> pd.DataFrame:
    """Combine decidento_score and financial_score into OSE_score (0–100)."""
    df = df.copy()
    for col in ["decidento_score", "financial_score"]:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")

    def minmax_norm(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        s_min = s.min()
        s_max = s.max()
        if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
            return pd.Series(0.5, index=s.index)
        return (s - s_min) / (s_max - s_min)

    df["decidento_norm"] = minmax_norm(df["decidento_score"])
    df["financial_norm"] = minmax_norm(df["financial_score"])

    total_weight = w_decidento + w_financial
    w_dec = w_decidento / total_weight
    w_fin = w_financial / total_weight

    df["OSE_raw"] = w_dec * df["decidento_norm"] + w_fin * df["financial_norm"]
    df["OSE_score"] = (df["OSE_raw"] * 100).round(2)
    return df


def _build_signal_counts(df_features: pd.DataFrame) -> pd.DataFrame:
    """Explode signals list to counts per signal code."""
    if "signals" not in df_features.columns:
        return pd.DataFrame(columns=["siren"])

    records = []
    for _, row in df_features[["siren", "signals"]].iterrows():
        if not isinstance(row["signals"], list):
            continue
        for item in row["signals"]:
            if isinstance(item, dict):
                rec = dict(item)
                rec["siren"] = row["siren"]
                records.append(rec)
    if not records:
        return pd.DataFrame(columns=["siren"])

    df_signals = pd.DataFrame(records)
    # Prepare + filter pre-t0 with shared utilities
    df_signals_prepared = signal_utils.prepare_signals(df_signals)
    if "publishedAt" in df_signals_prepared.columns and pd.api.types.is_datetime64_any_dtype(df_signals_prepared["publishedAt"]):
        # Align tz awareness with naive T0
        df_signals_prepared["publishedAt"] = df_signals_prepared["publishedAt"].dt.tz_localize(None)
    df_signals_pre_t0 = signal_utils.filter_pre_t0_signals(df_signals_prepared)
    df_signal_features = signal_utils.build_signal_features(df_signals_pre_t0)
    if df_signal_features.empty:
        return pd.DataFrame(columns=["siren"])

    # Align names with decidento weights (signal_<code>_count)
    rename_map = {col: col.replace("n_code_", "signal_") for col in df_signal_features.columns if col.startswith("n_code_")}
    df_signal_features = df_signal_features.rename(columns=rename_map)
    return df_signal_features


def _build_financial_panel(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create a normalized financial panel (siren, year, ca_final, resultat_final, effectif, capital_social, nbEtabSecondaire, age_entreprise).
    """
    if "siren" not in df_features.columns:
        return pd.DataFrame()

    df = df_features.copy()
    df["siren"] = df["siren"].astype(str)
    year_col = _first_existing(df, ["year"])
    if year_col:
        df["year"] = _parse_year(df[year_col])
    else:
        df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    ca_col = _first_existing(df, ["ca_final", "ca", "ca_bilan", "chiffre_d_affaires", "caConsolide", "caGroupe"])
    res_col = _first_existing(df, ["resultat_final", "resultat_net", "resultat_bilan", "resultat_exploitation", "resultatExploitation"])
    eff_col = _first_existing(df, ["effectif", "effectifConsolide", "effectifEstime"])
    cap_col = _first_existing(df, ["capital_social", "capital", "capitalSocial"])

    panel = pd.DataFrame({"siren": df["siren"], "year": df["year"]})
    panel["ca_final"] = df[ca_col] if ca_col else 0.0
    panel["resultat_final"] = df[res_col] if res_col else 0.0
    panel["effectif"] = df[eff_col] if eff_col else 0.0
    panel["capital_social"] = df[cap_col] if cap_col else 0.0

    if "nbEtabSecondaire" in df.columns:
        panel["nbEtabSecondaire"] = df["nbEtabSecondaire"]

    # Age from creation date if available
    if "dateCreationUniteLegale" in df.columns:
        creation_date = pd.to_datetime(df["dateCreationUniteLegale"], errors="coerce")
        panel["age_entreprise"] = ((T0 - creation_date) / pd.Timedelta(days=365.25)).round(1)
    else:
        panel["age_entreprise"] = np.nan

    # Filter pre-t0 if year is available
    if panel["year"].notna().any():
        panel = panel[panel["year"].isna() | (panel["year"] < T0.year)]

    return panel


def aggregate_financial_and_signal_scores(df_features: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    """
    Build financial+signal scores (latest pre-t0) and merge back into features.
    Writes a summary report to report_path.
    """
    if df_features.empty:
        return df_features

    panel = _build_financial_panel(df_features)
    if panel.empty:
        return df_features

    panel_scored = compute_financial_score(panel)

    # Aggregate to latest year per siren (pre-t0)
    panel_scored = panel_scored.sort_values(["siren", "year"])
    latest = panel_scored.groupby("siren").tail(1).copy()
    latest = latest.rename(columns={
        "financial_score": "financial_score_last",
        "growth_score": "growth_score_last",
        "profit_score": "profit_score_last",
        "ca_growth": "ca_growth_last",
        "effectif_growth": "effectif_growth_last",
        "margin_norm": "margin_norm_last",
        "capital_ratio": "capital_ratio_last",
        "year": "financial_year_last",
    })

    # Bring 3-year means if panel has multiple years
    tail3 = panel_scored.groupby("siren").tail(3)
    agg_3y = tail3.groupby("siren").agg(
        financial_score_3y_mean=("financial_score", "mean"),
        growth_score_3y_mean=("growth_score", "mean"),
        profit_score_3y_mean=("profit_score", "mean"),
    ).reset_index()

    latest = latest.merge(agg_3y, on="siren", how="left")

    # Signal counts
    signal_counts = _build_signal_counts(df_features)
    enriched = latest.merge(signal_counts, on="siren", how="left")
    # Base score used for OSE combination
    enriched["financial_score"] = enriched["financial_score_last"].fillna(0)

    # Decidento + OSE scores
    decidento = compute_decidento_score(enriched)
    combined = compute_total_ose_score(decidento)

    keep_cols = [
        "siren",
        "financial_score_last",
        "growth_score_last",
        "profit_score_last",
        "ca_growth_last",
        "effectif_growth_last",
        "margin_norm_last",
        "capital_ratio_last",
        "financial_year_last",
        "financial_score_3y_mean",
        "growth_score_3y_mean",
        "profit_score_3y_mean",
        "signal_score",
        "decidento_base_score",
        "decidento_score",
        "OSE_score",
    ]
    signal_cols = [c for c in decidento.columns if c.startswith("signal_") and c != "signal_score"]
    keep_cols.extend(signal_cols)

    available_keep_cols = [c for c in keep_cols if c in combined.columns]
    combined = combined[available_keep_cols]
    merged = df_features.merge(combined, on="siren", how="left")

    financial_score_series = merged["financial_score_last"] if "financial_score_last" in merged else pd.Series(dtype=float)
    summary = {
        "n_companies_scored": int(financial_score_series.notna().sum()),
        "financial_columns": [c for c in merged.columns if "financial_score" in c or "growth_score" in c or "profit_score" in c],
        "signal_columns": signal_cols,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  ✓ Financial & signal scores saved to {report_path}")
    return merged


