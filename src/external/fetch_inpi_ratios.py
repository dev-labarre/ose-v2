"""
Fetch external INPI ratios from data.economie.gouv.fr API.
Cache results to data/external/inpi_ratios.parquet.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Optional
import time
import json


def fetch_ratios_for_siren(siren: str, limit: int = 100, 
                           start_year: int = 2013, end_year: int = 2025,
                           max_retries: int = 3) -> List[Dict]:
    """
    Fetch INPI ratios for a single SIREN with retry logic for rate limiting.
    
    Args:
        siren: SIREN identifier (9 digits)
        limit: Maximum number of records to fetch
        start_year: Start year for date range
        end_year: End year for date range
        max_retries: Maximum number of retry attempts for rate limit errors
        
    Returns:
        List of ratio records
    """
    url = (
        f"https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/ratios_inpi_bce/records"
        f"?where=siren%20%3D%20%22{siren}%22"
        f"%20AND%20date_cloture_exercice%3A%5B%22{start_year}-01-01%22%20TO%20%22{end_year}-12-31%22%5D"
        f"&limit={limit}"
    )
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30)
            
            # Handle rate limiting (429)
            if r.status_code == 429:
                # Check for Retry-After header
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after) + 1  # Add 1 second buffer
                else:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = (2 ** attempt) + 1
                
                if attempt < max_retries - 1:
                    print(f"  ⚠️  Rate limited for SIREN {siren}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ⚠️  Rate limited for SIREN {siren} after {max_retries} attempts, skipping")
                    return []
            
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                print(f"  ⚠️  Error fetching ratios for SIREN {siren}: {e}")
                print(f"      Retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️  Error fetching ratios for SIREN {siren} after {max_retries} attempts: {e}")
                return []
        except Exception as e:
            print(f"  ⚠️  Unexpected error fetching ratios for SIREN {siren}: {e}")
            return []
    
    return []


def extract_ratio_fields(record: Dict) -> Optional[Dict]:
    """Extract relevant fields from API record."""
    try:
        fields = record.get("record", {}).get("fields", {})
        return {
            'siren': fields.get('siren'),
            'date_cloture_exercice': fields.get('date_cloture_exercice'),
            'ca': fields.get('ca'),
            'resultat_net': fields.get('resultat_net'),
            'effectif': fields.get('effectif'),
            'fonds_propres': fields.get('fonds_propres'),
            'dettes_financieres': fields.get('dettes_financieres'),
            'bfr': fields.get('bfr'),
            'ca_export': fields.get('ca_export'),
            'valeur_ajoutee': fields.get('valeur_ajoutee'),
            'excedent_brut_exploitation': fields.get('excedent_brut_exploitation'),
            'resultat_exploitation': fields.get('resultat_exploitation'),
            'charges_personnel': fields.get('charges_personnel'),
            'impots_taxes': fields.get('impots_taxes'),
        }
    except Exception as e:
        print(f"  ⚠️  Error extracting fields from record: {e}")
        return None


def fetch_all_ratios(siren_list: List[str], output_path: Path, 
                    start_year: int = 2013, end_year: int = 2025,
                    rate_limit_delay: float = 2.0):
    """
    Fetch ratios for all SIRENs and save to parquet.
    
    Args:
        siren_list: List of SIREN identifiers
        output_path: Path to save parquet file
        start_year: Start year for date range
        end_year: End year for date range
        rate_limit_delay: Delay between requests (seconds) - increased to 2s to avoid rate limiting
    """
    print("="*80)
    print("FETCHING EXTERNAL INPI RATIOS")
    print("="*80)
    print(f"\nFetching ratios for {len(siren_list)} SIRENs...")
    print(f"Date range: {start_year}-{end_year}")
    print(f"Rate limit delay: {rate_limit_delay}s between requests")
    print(f"Output: {output_path}\n")
    
    all_rows = []
    failed_sirens = []
    
    for i, siren in enumerate(siren_list, start=1):
        siren = str(siren).zfill(9)  # Ensure 9 digits
        
        # Add delay BEFORE request to respect rate limits
        if i > 1:  # Don't delay before first request
            time.sleep(rate_limit_delay)
        
        records = fetch_ratios_for_siren(siren, start_year=start_year, end_year=end_year)
        
        if not records:
            failed_sirens.append(siren)
        
        for record in records:
            extracted = extract_ratio_fields(record)
            if extracted:
                all_rows.append(extracted)
        
        if i % 50 == 0 or i == len(siren_list):
            print(f"  Processed {i}/{len(siren_list)} SIRENs — {len(all_rows)} ratio records")
            if failed_sirens:
                print(f"    ⚠️  {len(failed_sirens)} SIRENs failed (rate limited or errors)")
    
    if not all_rows:
        print("\n  ⚠️  No ratio records fetched")
        if failed_sirens:
            print(f"  All {len(failed_sirens)} SIRENs failed due to rate limiting")
            print("  💡 Tip: Try running again later, or reduce the number of SIRENs")
        return
    
    df_ratios = pd.DataFrame(all_rows)
    
    # Parse date
    if 'date_cloture_exercice' in df_ratios.columns:
        df_ratios['date_cloture_exercice'] = pd.to_datetime(
            df_ratios['date_cloture_exercice'], errors='coerce'
        )
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ratios.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved {len(df_ratios)} ratio records to {output_path}")
    print(f"  Unique SIRENs: {df_ratios['siren'].nunique()}")
    if 'date_cloture_exercice' in df_ratios.columns:
        print(f"  Date range: {df_ratios['date_cloture_exercice'].min()} to {df_ratios['date_cloture_exercice'].max()}")
    
    if failed_sirens:
        print(f"\n  ⚠️  {len(failed_sirens)} SIRENs failed to fetch (rate limited)")
        print(f"  Success rate: {len(siren_list) - len(failed_sirens)}/{len(siren_list)} ({100 * (len(siren_list) - len(failed_sirens)) / len(siren_list):.1f}%)")


def main():
    """Main function to fetch ratios."""
    project_root = Path(__file__).resolve().parents[2]
    
    # Try to load SIREN list from features first, fallback to raw data
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    raw_data_path = project_root / 'data' / 'raw_json' / '01_company_basic_info.json'
    
    siren_list = []
    
    if features_path.exists():
        # Use features if available
        df_features = pd.read_parquet(features_path)
        siren_list = df_features['siren'].unique().tolist()
        print(f"  ✓ Loaded {len(siren_list)} SIRENs from features.parquet")
    elif raw_data_path.exists():
        # Fallback to raw data
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        df_raw = pd.DataFrame(raw_data)
        if 'siren' in df_raw.columns:
            siren_list = df_raw['siren'].unique().tolist()
            print(f"  ✓ Loaded {len(siren_list)} SIRENs from raw data")
        else:
            print(f"⚠️  No 'siren' column found in {raw_data_path}")
            return
    else:
        print(f"⚠️  No data source found. Expected:")
        print(f"    - {features_path}")
        print(f"    - {raw_data_path}")
        print("  Run 'make extract' first to generate raw data")
        return
    
    if not siren_list:
        print("⚠️  No SIRENs found to fetch ratios for")
        return
    
    output_path = project_root / 'data' / 'external' / 'inpi_ratios.parquet'
    
    # Check if already exists
    if output_path.exists():
        print(f"⚠️  Ratios file already exists: {output_path}")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("  Skipping fetch")
            return
    
    fetch_all_ratios(siren_list, output_path)


if __name__ == "__main__":
    main()
