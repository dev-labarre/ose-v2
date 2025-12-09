#!/usr/bin/env python3
"""
Extract KPI data from the Decidento dataset.
"""

import json
from pathlib import Path

from base_loader import load_source


def main():
    records = []
    for comp in load_source():
        kpi = comp.get("last_kpi") or {}
        company_name = comp.get("raison_sociale") or comp.get("raison_sociale_keyword") or ""
        year = kpi.get("annee") or kpi.get("year")
        record = {
            "company_name": company_name,
            "siren": comp.get("siren"),
            "siret": str(comp.get("siret_siege", "")),
            "year": year,
            "fonds_propres": kpi.get("fonds_propres"),
            "ca_france": kpi.get("ca_france"),
            "date_cloture_exercice": kpi.get("date_cloture_exercice"),
            "duree_exercice": kpi.get("duree_exercice"),
            "salaires_traitements": kpi.get("salaires_traitements"),
            "charges_financieres": kpi.get("charges_financieres"),
            "impots_taxes": kpi.get("impots_taxes"),
            "ca_bilan": kpi.get("ca_bilan"),
            "resultat_exploitation": kpi.get("resultat_exploitation"),
            "dotations_amortissements": kpi.get("dotations_amortissements"),
            "capital_social": kpi.get("capital_social"),
            "code_confidentialite": kpi.get("code_confidentialite"),
            "resultat_bilan": kpi.get("resultat_bilan"),
            "annee": year,
            "effectif": kpi.get("effectif"),
            "effectif_sous_traitance": kpi.get("effectif_sous_traitance"),
            "filiales_participations": kpi.get("filiales_participations"),
            "evolution_ca": kpi.get("evolution_ca"),
            "subventions_investissements": kpi.get("subventions_investissements"),
            "ca_export": kpi.get("ca_export"),
            "evolution_effectif": kpi.get("evolution_effectif"),
            "participation_bilan": kpi.get("participation_bilan"),
            "ca_consolide": kpi.get("ca_consolide"),
            "resultat_net_consolide": kpi.get("resultat_net_consolide"),
        }
        records.append(record)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "07_kpi_data.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exported {len(records):,} KPI records to {output_file}")


if __name__ == "__main__":
    main()
