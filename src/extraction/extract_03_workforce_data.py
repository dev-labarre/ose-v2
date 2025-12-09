#!/usr/bin/env python3
"""
Extract workforce data from the Decidento dataset.
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
                "effectif": kpi.get("effectif"),
                "effectifConsolide": kpi.get("effectif_consolide"),
                "effectifEstime": kpi.get("effectif_estime") or comp.get("effectif_estime"),
                "effectifGroupe": kpi.get("effectif_groupe") or comp.get("effectif_groupe"),
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "03_workforce_data.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} workforce records to {output_file}")


if __name__ == "__main__":
    main()
