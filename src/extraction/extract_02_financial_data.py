#!/usr/bin/env python3
"""
Extract consolidated financial metrics from the Decidento dataset.
"""

import json
from pathlib import Path

from base_loader import load_source


def main():
    records = []
    for comp in load_source():
        kpi = comp.get("last_kpi") or {}
        company_name = comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or ""
        records.append(
            {
                "company_name": company_name,
                "siren": str(comp.get("siren", "")),
                "siret": str(comp.get("siret_siege", "")),
                "caConsolide": kpi.get("ca_consolide"),
                "caGroupe": kpi.get("ca_groupe") or comp.get("ca_groupe"),
                "resultatExploitation": kpi.get("resultat_exploitation"),
                "dateConsolide": kpi.get("date_cloture_exercice"),
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "02_financial_data.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} financial records to {output_file}")


if __name__ == "__main__":
    main()
