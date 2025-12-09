#!/usr/bin/env python3
"""
Extract signals from projects in the separate/ folder.
"""

import json
from pathlib import Path


def extract_signal_from_project(project):
    """Extract signal record from project."""
    all_companies = project.get('allCompanies', [])
    main_company = None
    for comp in all_companies:
        if comp.get('isMain', False):
            main_company = comp
            break
    if not main_company and all_companies:
        main_company = all_companies[0]
    
    if not main_company:
        return None
    
    continent = project.get('continent', [])
    country = project.get('country', [])
    departement = project.get('departement', [])
    
    companies_count = len(project.get('allCompanies', []))
    sirets_count = len(project.get('sirets', []))
    
    return {
        'company_name': main_company.get('name', ''),
        'siren': str(main_company.get('siren', '')),
        'siret': str(main_company.get('siret', '')) if main_company.get('siret') else '',
        'continent': continent if isinstance(continent, list) else [],
        'country': country if isinstance(country, list) else [],
        'departement': departement if isinstance(departement, list) else [],
        'publishedAt': project.get('publishedAt'),
        'isMain': main_company.get('isMain', False),
        'type': project.get('type', {}),
        'createdAt': project.get('createdAt'),
        'companies_count': companies_count,
        'sirets_count': sirets_count
    }


def main():
    # Path to source file
    project_root = Path(__file__).resolve().parents[2]
    source_file = project_root / "data" / "source" / "separate" / "agro_alim_projects.json"
    
    if not source_file.exists():
        print(f"⚠️  Source file not found: {source_file}")
        print("Writing empty list to maintain pipeline compatibility.")
        output_dir = project_root / "data" / "raw_json"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "08_signals.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return
    
    # Load projects
    print(f"📥 Loading projects from: {source_file}")
    with source_file.open("r", encoding="utf-8") as f:
        projects = json.load(f)
    print(f"  ✓ Loaded {len(projects):,} projects")
    
    # Extract signals
    print("\n📊 Extracting signals...")
    records = []
    for i, project in enumerate(projects):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,} projects...")
        signal = extract_signal_from_project(project)
        if signal:
            records.append(signal)
    
    # Save output
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "08_signals.json"
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Exported {len(records):,} signal records to {output_file}")


if __name__ == "__main__":
    main()
