#!/usr/bin/env python3
"""
Extract contact metrics from the Decidento dataset.
"""

import json
from pathlib import Path

from base_loader import load_source


def main():
    records = []
    for comp in load_source():
        company_name = comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or ""
        records.append(
            {
                "company_name": company_name,
                "siren": str(comp.get("siren", "")),
                "siret": str(comp.get("siret_siege", "")),
                "nombre_societeinfo_contact_email": comp.get("nombre_societeinfo_contact_email"),
                "nombre_societeinfo_contact_linkedin_url": comp.get(
                    "nombre_societeinfo_contact_linkedin_url"
                ),
                "nombre_societeinfo_contact_pro": comp.get("nombre_societeinfo_contact_pro"),
                "nombre_societeinfo_contact_perso": comp.get("nombre_societeinfo_contact_perso"),
                "nombre_societeinfo_contact_max2": comp.get("nombre_societeinfo_contact_max2"),
                "nombre_societeinfo_contact_max5": comp.get("nombre_societeinfo_contact_max5"),
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "06_contact_metrics.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} contact metric records to {output_file}")


if __name__ == "__main__":
    main()
