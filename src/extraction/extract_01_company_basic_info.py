#!/usr/bin/env python3
"""
Extract company basic info from the Decidento line-delimited dataset.
"""

import json
from pathlib import Path

from base_loader import load_source, normalize_dept, to_timestamp


def main():
    records = []
    for comp in load_source():
        last_modified = comp.get("last_modified")
        ts = to_timestamp(last_modified)
        resume = comp.get("resume_activite") or comp.get("descriptif_activity") or ""
        company_name = comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or ""

        records.append(
            {
                "company_name": company_name,
                "siren": str(comp.get("siren", "")),
                "siret": str(comp.get("siret_siege", "")),
                "departement": normalize_dept(comp.get("departement")),
                "resume_activite": resume,
                "raison_sociale_keyword": comp.get("raison_sociale_keyword", ""),
                "raison_sociale": comp.get("raison_sociale", ""),
                "last_modified": last_modified,
                "processedAt": ts,
                "updatedAt": ts,
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "01_company_basic_info.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} company basic records to {output_file}")


if __name__ == "__main__":
    main()
