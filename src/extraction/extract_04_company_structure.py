#!/usr/bin/env python3
"""
Extract company structure metrics from the Decidento dataset.
"""

import json
from pathlib import Path

from base_loader import load_source


def main():
    records = []
    for comp in load_source():
        enseignes = comp.get("enseignes")
        nb_marques = len(enseignes) if isinstance(enseignes, list) else None
        records.append(
            {
                "company_name": comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or "",
                "siren": str(comp.get("siren", "")),
                "siret": str(comp.get("siret_siege", "")),
                "nbFilialesDirectes": comp.get("nb_filiales_directes"),  # not provided: remains None
                "nbEtabSecondaire": comp.get("nombre_etablissements_secondaires_actifs"),
                "nbMarques": nb_marques,
                "hasGroupOwner": bool(comp.get("group_owner_siren")),
                "appartient_groupe": comp.get("appartient_groupe"),
                "nombre_etablissements_secondaires_inactifs": comp.get(
                    "nombre_etablissements_secondaires_inactifs"
                ),
            }
        )

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "04_company_structure.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} company structure records to {output_file}")


if __name__ == "__main__":
    main()
