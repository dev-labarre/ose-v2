#!/usr/bin/env python3
"""
Extract classification flags from the Decidento dataset.
"""

import json
from pathlib import Path

from base_loader import load_source


def main():
    records = []
    for comp in load_source():
        enseignes = comp.get("enseignes")
        nb_marques = len(enseignes) if isinstance(enseignes, list) else 0
        company_name = comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or ""
        records.append(
            {
                "company_name": company_name,
                "siren": str(comp.get("siren", "")),
                "siret": str(comp.get("siret_siege", "")),
                "startup": comp.get("startup"),
                "radiee": comp.get("radiee"),
                "entreprise_b2b": comp.get("entreprise_b2b"),
                "entreprise_b2c": comp.get("entreprise_b2c"),
                "fintech": comp.get("entreprise_fintech") or comp.get("fintech"),
                "cac40": comp.get("cac40"),
                "entreprise_familiale": comp.get("entreprise_familiale"),
                "entreprise_familiale_ter": str(comp.get("entreprise_familiale"))
                if comp.get("entreprise_familiale") is not None
                else None,
                "filtre_levee_fond": comp.get("filtre_levee_fond"),
                "flag_type_entreprise": comp.get("flag_type_entreprise"),
                "hasMarques": nb_marques > 0,
                "hasESV1Contacts": comp.get("has_esv1_contacts"),
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "05_classification_flags.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} classification flag records to {output_file}")


if __name__ == "__main__":
    main()
