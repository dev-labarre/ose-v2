#!/usr/bin/env python3
"""
Extract articles from the separate/ folder.
"""

import json
from pathlib import Path


def extract_article_record(article):
    """Extract article record with company info."""
    all_companies = article.get('all_companies', [])
    if not all_companies:
        all_companies = article.get('allCompanies', []) or article.get('companies', [])
    
    main_company = None
    if all_companies and len(all_companies) > 0:
        comp = all_companies[0]
        if isinstance(comp, dict):
            main_company = comp
        elif isinstance(comp, int):
            main_company = {'id': comp, 'name': '', 'siren': '', 'siret': ''}
    
    if not main_company:
        main_company = {'name': '', 'siren': '', 'siret': ''}
    
    return {
        'company_name': main_company.get('name', ''),
        'siren': str(main_company.get('siren', '')),
        'siret': str(main_company.get('siret', '')) if main_company.get('siret') else '',
        'title': article.get('title', ''),
        'publishedAt': article.get('publishedAt'),
        'author': article.get('author', {}),
        'signalsStatus': article.get('signalsStatus', []),
        'signalsType': article.get('signalsType', []),
        'country': article.get('country', {}),
        'sectors': article.get('sectors', []),
        'cities': article.get('cities', []),
        'sources': article.get('sources', []),
        'all_companies_count': len(all_companies) if all_companies else 0
    }


def main():
    # Path to source file
    project_root = Path(__file__).resolve().parents[2]
    source_file = project_root / "data" / "source" / "separate" / "agro_alim_articles.json"
    
    if not source_file.exists():
        print(f"⚠️  Source file not found: {source_file}")
        print("Writing empty list to maintain pipeline compatibility.")
        output_dir = project_root / "data" / "raw_json"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "09_articles.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return
    
    # Load articles
    print(f"📥 Loading articles from: {source_file}")
    with source_file.open("r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  ✓ Loaded {len(articles):,} articles")
    
    # Extract article records
    print("\n📊 Extracting article records...")
    records = []
    for i, article in enumerate(articles):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,} articles...")
        record = extract_article_record(article)
        if record:
            records.append(record)
    
    # Save output
    output_dir = project_root / "data" / "raw_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "09_articles.json"
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Exported {len(records):,} article records to {output_file}")


if __name__ == "__main__":
    main()
