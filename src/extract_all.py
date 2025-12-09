#!/usr/bin/env python3
"""
Unified extraction script for Business Opportunity Classifier.
Extracts all 9 datasets from source files and saves to data/raw_json/.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Expected column orders from the legacy/OLD DATA structure. Using an explicit
# mapping lets us preserve column shape even when new fields are missing in
# the current source.
EXPECTED_COLUMNS = {
    'company_basic_info': [
        'company_name',
        'siren',
        'siret',
        'departement',
        'resume_activite',
        'raison_sociale_keyword',
        'raison_sociale',
        'last_modified',
        'processedAt',
        'updatedAt',
    ],
    'financial_data': [
        'company_name',
        'siren',
        'siret',
        'caConsolide',
        'caGroupe',
        'resultatExploitation',
        'dateConsolide',
        'kpi_2025_capital_social',
        'kpi_2025_evolution_ca',
        'kpi_2023_ca_france',
        'kpi_2023_ca_bilan',
        'kpi_2023_resultat_exploitation',
        'kpi_2023_capital_social',
        'kpi_2023_resultat_bilan',
        'kpi_2022_ca_france',
        'kpi_2022_ca_bilan',
        'kpi_2022_resultat_exploitation',
        'kpi_2022_capital_social',
        'kpi_2022_resultat_bilan',
        'kpi_2021_ca_france',
        'kpi_2021_ca_bilan',
        'kpi_2021_resultat_exploitation',
        'kpi_2021_capital_social',
        'kpi_2021_resultat_bilan',
        'kpi_2020_ca_france',
        'kpi_2020_ca_bilan',
        'kpi_2020_resultat_exploitation',
        'kpi_2020_capital_social',
        'kpi_2020_resultat_bilan',
        'kpi_2019_ca_france',
        'kpi_2019_ca_bilan',
        'kpi_2019_resultat_exploitation',
        'kpi_2019_capital_social',
        'kpi_2019_resultat_bilan',
        'kpi_2018_ca_bilan',
        'kpi_2018_resultat_exploitation',
        'kpi_2018_capital_social',
        'kpi_2018_resultat_bilan',
        'kpi_2017_ca_france',
        'kpi_2017_ca_bilan',
        'kpi_2017_resultat_exploitation',
        'kpi_2017_capital_social',
        'kpi_2017_resultat_bilan',
        'kpi_2016_ca_bilan',
        'kpi_2016_resultat_bilan',
        'kpi_2015_ca_bilan',
        'kpi_2015_resultat_bilan',
        'kpi_2014_ca_bilan',
        'kpi_2014_resultat_bilan',
        'kpi_2020_ca_export',
        'kpi_2018_ca_france',
        'kpi_2023_ca_export',
        'kpi_2016_ca_france',
        'kpi_2016_resultat_exploitation',
        'kpi_2016_ca_export',
        'kpi_2016_capital_social',
        'kpi_2015_capital_social',
        'kpi_2013_ca_bilan',
        'kpi_2013_resultat_bilan',
        'kpi_2017_ca_export',
        'kpi_2014_capital_social',
        'kpi_2024_capital_social',
        'kpi_2024_evolution_ca',
        'kpi_2022_ca_export',
        'kpi_2021_ca_export',
        'kpi_2019_ca_export',
        'kpi_2018_ca_export',
        'kpi_2015_ca_france',
        'kpi_2015_resultat_exploitation',
        'kpi_2015_ca_export',
        'kpi_2022_evolution_ca',
        'kpi_2013_ca_france',
        'kpi_2013_resultat_exploitation',
        'kpi_2013_capital_social',
        'kpi_2012_ca_bilan',
        'kpi_2012_resultat_bilan',
        'kpi_2011_ca_bilan',
        'kpi_2011_resultat_bilan',
        'kpi_2024_resultat_bilan',
        'kpi_2024_ca_france',
        'kpi_2024_ca_bilan',
        'kpi_2024_resultat_exploitation',
        'kpi_2010_capital_social',
        'kpi_2012_capital_social',
        'kpi_2025_ca_bilan',
        'kpi_2025_resultat_exploitation',
        'kpi_2025_resultat_bilan',
        'kpi_2024_ca_export',
        'kpi_2014_ca_france',
        'kpi_2014_resultat_exploitation',
        'kpi_2014_ca_export',
        'kpi_2011_capital_social',
        'kpi_2012_ca_france',
        'kpi_2012_resultat_exploitation',
        'kpi_2023_evolution_ca',
        'kpi_2025_ca_france',
        'kpi_2025_ca_export',
        'kpi_2013_ca_export',
        'kpi_2019_ca_consolide',
        'kpi_2019_resultat_net_consolide',
        'kpi_2018_ca_consolide',
        'kpi_2018_resultat_net_consolide',
        'kpi_2017_ca_consolide',
        'kpi_2016_ca_consolide',
        'kpi_2016_resultat_net_consolide',
        'kpi_2023_ca_consolide',
        'kpi_2023_resultat_net_consolide',
        'kpi_2022_ca_consolide',
        'kpi_2022_resultat_net_consolide',
        'kpi_2021_ca_consolide',
        'kpi_2021_resultat_net_consolide',
        'kpi_2017_resultat_net_consolide',
    ],
    'workforce_data': [
        'company_name',
        'siren',
        'siret',
        'effectif',
        'effectifConsolide',
        'effectifEstime',
        'effectifGroupe',
    ],
    'company_structure': [
        'company_name',
        'siren',
        'siret',
        'nbFilialesDirectes',
        'nbEtabSecondaire',
        'nbMarques',
        'hasGroupOwner',
        'appartient_groupe',
        'nombre_etablissements_secondaires_inactifs',
    ],
    'classification_flags': [
        'company_name',
        'siren',
        'siret',
        'startup',
        'radiee',
        'entreprise_b2b',
        'entreprise_b2c',
        'fintech',
        'cac40',
        'entreprise_familiale',
        'entreprise_familiale_ter',
        'filtre_levee_fond',
        'flag_type_entreprise',
        'hasMarques',
        'hasESV1Contacts',
    ],
    'contact_metrics': [
        'company_name',
        'siren',
        'siret',
        'nombre_societeinfo_contact_email',
        'nombre_societeinfo_contact_linkedin_url',
        'nombre_societeinfo_contact_pro',
        'nombre_societeinfo_contact_perso',
        'nombre_societeinfo_contact_max2',
        'nombre_societeinfo_contact_max5',
    ],
    'kpi_data': [
        'company_name',
        'siren',
        'siret',
        'year',
        'fonds_propres',
        'ca_france',
        'date_cloture_exercice',
        'duree_exercice',
        'salaires_traitements',
        'charges_financieres',
        'impots_taxes',
        'ca_bilan',
        'resultat_exploitation',
        'dotations_amortissements',
        'capital_social',
        'code_confidentialite',
        'resultat_bilan',
        'annee',
        'effectif',
        'effectif_sous_traitance',
        'filiales_participations',
        'evolution_ca',
        'subventions_investissements',
        'ca_export',
        'evolution_effectif',
        'participation_bilan',
        'ca_consolide',
        'resultat_net_consolide',
    ],
    'signals': [
        'company_name',
        'siren',
        'siret',
        'continent',
        'country',
        'departement',
        'publishedAt',
        'isMain',
        'type',
        'createdAt',
        'companies_count',
        'sirets_count',
    ],
    'articles': [
        'company_name',
        'siren',
        'siret',
        'title',
        'publishedAt',
        'author',
        'signalsStatus',
        'signalsType',
        'country',
        'sectors',
        'cities',
        'sources',
        'all_companies_count',
    ],
}


def extract_year_from_date(date_str):
    """Extract year from date string."""
    if not date_str:
        return None
    try:
        if 'T' in str(date_str):
            return int(str(date_str).split('T')[0].split('-')[0])
        elif '-' in str(date_str):
            return int(str(date_str).split('-')[0])
        return None
    except:
        return None


def extract_company_basic_info(company):
    """Extract basic company information."""
    dept = company.get('department', {})
    if isinstance(dept, dict):
        dept_id = str(dept.get('id', '')) if dept.get('id') else ''
        if len(dept_id) == 1:
            dept_id = '0' + dept_id
    else:
        dept_id = ''
    
    resume_activite = company.get('activityLight', '') or company.get('activity', '')
    
    updated_at = company.get('updatedAt', '')
    
    processed_at = None
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            processed_at = int(dt.timestamp())
        except:
            processed_at = None
    
    updated_at_ts = None
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            updated_at_ts = int(dt.timestamp())
        except:
            updated_at_ts = None
    
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'departement': dept_id,
        'resume_activite': resume_activite,
        'raison_sociale_keyword': company.get('socialName', ''),
        'raison_sociale': company.get('socialName', ''),
        'last_modified': updated_at,
        'processedAt': processed_at,
        'updatedAt': updated_at_ts
    }


def extract_financial_data(company):
    """Extract financial information."""
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'caConsolide': company.get('caConsolide'),
        'caGroupe': company.get('caGroupe'),
        'resultatExploitation': company.get('resultatExploitation'),
        'dateConsolide': company.get('dateConsolide')
    }


def extract_workforce_data(company):
    """Extract workforce information."""
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'effectif': company.get('effectif'),
        'effectifConsolide': company.get('effectifConsolide'),
        'effectifEstime': company.get('effectifEstime'),
        'effectifGroupe': company.get('effectifGroupe')
    }


def extract_company_structure(company):
    """Extract company structure information."""
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'nbFilialesDirectes': company.get('nbFilialesDirectes', 0),
        'nbEtabSecondaire': 0,
        'nbMarques': company.get('nbMarques') if company.get('hasMarques') else 0,
        'hasGroupOwner': company.get('hasGroupOwner', False),
        'appartient_groupe': company.get('hasGroupOwner', False),
        'nombre_etablissements_secondaires_inactifs': 0
    }


def extract_classification_flags(company):
    """Extract classification flags."""
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'startup': company.get('startup', False),
        'radiee': company.get('radiate', False),
        'entreprise_b2b': company.get('bToB', False),
        'entreprise_b2c': company.get('bToC', False),
        'fintech': company.get('entreprise_fintech', False),
        'cac40': company.get('cac40', False),
        'entreprise_familiale': company.get('entreprise_familiale', False),
        'entreprise_familiale_ter': company.get('entreprise_familiale_ter', 'false'),
        'filtre_levee_fond': False,
        'flag_type_entreprise': None,
        'hasMarques': company.get('hasMarques', False),
        'hasESV1Contacts': company.get('hasESV1Contacts', False)
    }


def extract_contact_metrics(company):
    """Extract contact metrics."""
    return {
        'company_name': company.get('socialName', ''),
        'siren': str(company.get('siren', '')),
        'siret': str(company.get('siret', '')) if company.get('siret') else '',
        'nombre_societeinfo_contact_email': None,
        'nombre_societeinfo_contact_linkedin_url': None,
        'nombre_societeinfo_contact_pro': None,
        'nombre_societeinfo_contact_perso': None,
        'nombre_societeinfo_contact_max2': None,
        'nombre_societeinfo_contact_max5': None
    }


def extract_kpi_data(company):
    """Extract KPI data. Only creates records with valid years."""
    records = []
    
    company_name = company.get('socialName', '')
    siren = company.get('siren', '')
    siret = company.get('siret', '')
    
    # Extract company-level fields for fallback
    ca_bilan = company.get('caBilan')
    ca_consolide = company.get('caConsolide')
    fonds_propres = company.get('fondsPropres')
    resultat_exploitation = company.get('resultatExploitation')
    resultat_net = company.get('resultatNet')
    effectif = company.get('effectif')
    date_cloture = company.get('dateCloture')
    
    year_from_date = extract_year_from_date(date_cloture)
    kpis = company.get('kpis', [])
    
    if kpis and isinstance(kpis, list) and len(kpis) > 0:
        for kpi in kpis:
            if not isinstance(kpi, dict):
                continue
            year = kpi.get('year')
            kpi_effectif = kpi.get('effectif')
            record_year = year if year else year_from_date
            
            # FIX: Skip records without valid year
            if not record_year:
                continue
            
            record = {
                'company_name': company_name,
                'siren': int(siren) if siren and str(siren).isdigit() else None,
                'siret': str(siret) if siret else None,
                'year': int(record_year),
                'fonds_propres': float(fonds_propres) if fonds_propres is not None else None,
                'ca_france': None,
                'date_cloture_exercice': date_cloture if date_cloture else None,
                'duree_exercice': None,
                'salaires_traitements': None,
                'charges_financieres': None,
                'impots_taxes': None,
                'ca_bilan': float(ca_bilan) if ca_bilan is not None else None,
                'resultat_exploitation': float(resultat_exploitation) if resultat_exploitation is not None else None,
                'dotations_amortissements': None,
                'capital_social': None,
                'code_confidentialite': None,
                'resultat_bilan': float(resultat_net) if resultat_net is not None else None,
                'annee': int(record_year),
                'effectif': float(kpi_effectif) if kpi_effectif is not None else (float(effectif) if effectif is not None else None),
                'effectif_sous_traitance': None,
                'filiales_participations': None,
                'evolution_ca': None,
                'subventions_investissements': None,
                'ca_export': None,
                'evolution_effectif': None,
                'participation_bilan': None,
                'ca_consolide': float(ca_consolide) if ca_consolide is not None else None,
                'resultat_net_consolide': None,
            }
            records.append(record)
    else:
        # FIX: Skip records without valid year
        if not year_from_date:
            return records  # Return empty list if no year available
        
        record = {
            'company_name': company_name,
            'siren': int(siren) if siren and str(siren).isdigit() else None,
            'siret': str(siret) if siret else None,
            'year': int(year_from_date),
            'fonds_propres': float(fonds_propres) if fonds_propres is not None else None,
            'ca_france': None,
            'date_cloture_exercice': date_cloture if date_cloture else None,
            'duree_exercice': None,
            'salaires_traitements': None,
            'charges_financieres': None,
            'impots_taxes': None,
            'ca_bilan': float(ca_bilan) if ca_bilan is not None else None,
            'resultat_exploitation': float(resultat_exploitation) if resultat_exploitation is not None else None,
            'dotations_amortissements': None,
            'capital_social': None,
            'code_confidentialite': None,
            'resultat_bilan': float(resultat_net) if resultat_net is not None else None,
            'annee': int(year_from_date),
            'effectif': float(effectif) if effectif is not None else None,
            'effectif_sous_traitance': None,
            'filiales_participations': None,
            'evolution_ca': None,
            'subventions_investissements': None,
            'ca_export': None,
            'evolution_effectif': None,
            'participation_bilan': None,
            'ca_consolide': float(ca_consolide) if ca_consolide is not None else None,
            'resultat_net_consolide': None,
        }
        records.append(record)
    
    return records


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


def save_extraction(output_file, records, expected_columns, dataset_name):
    """Save extracted records to JSON file."""
    if not records:
        print(f"  ⚠️  No records extracted for {dataset_name}")
        return False
    
    df = pd.DataFrame(records)
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    df = df[expected_columns]
    
    output_data = df.to_dict('records')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  ✓ {dataset_name}: {len(records):,} records exported")
    return True


def main():
    """Main extraction function."""
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / 'data' / 'source'
    output_dir = project_root / 'data' / 'raw_json'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OSE - UNIFIED DATA EXTRACTION")
    print("="*80)
    print(f"\nSource directory: {source_dir}")
    print(f"Output directory: {output_dir}\n")
    
    companies_file = source_dir / 'agro_alim_companies.json'
    articles_file = source_dir / 'agro_alim_articles.json'
    projects_file = source_dir / 'agro_alim_projects.json'
    
    companies = []
    articles = []
    projects = []
    
    if companies_file.exists():
        print(f"📥 Loading companies from: {companies_file}")
        try:
            with open(companies_file, 'r', encoding='utf-8') as f:
                companies = json.load(f)
            print(f"  ✓ Loaded {len(companies):,} companies\n")
        except Exception as e:
            print(f"  ⚠️  Error loading companies: {e}\n")
    else:
        print(f"  ⚠️  Companies file not found: {companies_file}\n")
    
    if articles_file.exists():
        print(f"📥 Loading articles from: {articles_file}")
        try:
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"  ✓ Loaded {len(articles):,} articles\n")
        except Exception as e:
            print(f"  ⚠️  Error loading articles: {e}\n")
    else:
        print(f"  ⚠️  Articles file not found: {articles_file}\n")
    
    if projects_file.exists():
        print(f"📥 Loading projects from: {projects_file}")
        try:
            with open(projects_file, 'r', encoding='utf-8') as f:
                projects = json.load(f)
            print(f"  ✓ Loaded {len(projects):,} projects\n")
        except Exception as e:
            print(f"  ⚠️  Error loading projects: {e}\n")
    else:
        print(f"  ⚠️  Projects file not found: {projects_file}\n")
    
    if companies:
        print("="*80)
        print("EXTRACTING DATASETS FROM COMPANIES")
        print("="*80)
        
        print("\n1. Extracting company basic info...")
        records = []
        for i, company in enumerate(companies):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} companies...")
            records.append(extract_company_basic_info(company))
        save_extraction(
            output_dir / '01_company_basic_info.json',
            records,
            EXPECTED_COLUMNS['company_basic_info'],
            'Company Basic Info'
        )
        
        print("\n2. Extracting financial data...")
        records = []
        for company in companies:
            records.append(extract_financial_data(company))
        save_extraction(
            output_dir / '02_financial_data.json',
            records,
            EXPECTED_COLUMNS['financial_data'],
            'Financial Data'
        )
        
        print("\n3. Extracting workforce data...")
        records = []
        for company in companies:
            records.append(extract_workforce_data(company))
        save_extraction(
            output_dir / '03_workforce_data.json',
            records,
            EXPECTED_COLUMNS['workforce_data'],
            'Workforce Data'
        )
        
        print("\n4. Extracting company structure...")
        records = []
        for company in companies:
            records.append(extract_company_structure(company))
        save_extraction(
            output_dir / '04_company_structure.json',
            records,
            EXPECTED_COLUMNS['company_structure'],
            'Company Structure'
        )
        
        print("\n5. Extracting classification flags...")
        records = []
        for company in companies:
            records.append(extract_classification_flags(company))
        save_extraction(
            output_dir / '05_classification_flags.json',
            records,
            EXPECTED_COLUMNS['classification_flags'],
            'Classification Flags'
        )
        
        print("\n6. Extracting contact metrics...")
        records = []
        for company in companies:
            records.append(extract_contact_metrics(company))
        save_extraction(
            output_dir / '06_contact_metrics.json',
            records,
            EXPECTED_COLUMNS['contact_metrics'],
            'Contact Metrics'
        )
        
        print("\n7. Extracting KPI data...")
        records = []
        skipped_no_year = 0
        for i, company in enumerate(companies):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} companies, extracted {len(records):,} records, skipped {skipped_no_year:,} without year...")
            kpi_records = extract_kpi_data(company)
            if len(kpi_records) == 0:
                skipped_no_year += 1
            records.extend(kpi_records)
        # Additional safety check - filter out any records with null years
        if records:
            df = pd.DataFrame(records)
            before_filter = len(df)
            df = df[df['year'].notna()]
            after_filter = len(df)
            if before_filter != after_filter:
                print(f"  ⚠️  Filtered out {before_filter - after_filter} records with null years (safety check)")
                records = df.to_dict('records')
            else:
                # Ensure year and annee are proper integer types
                df['year'] = df['year'].astype('Int64')
                df['annee'] = df['annee'].astype('Int64')
                records = df.to_dict('records')
        
        if skipped_no_year > 0:
            print(f"  ℹ️  Companies skipped (no valid year): {skipped_no_year:,}")
        
        save_extraction(
            output_dir / '07_kpi_data.json',
            records,
            EXPECTED_COLUMNS['kpi_data'],
            'KPI Data'
        )
    
    if projects:
        print("\n" + "="*80)
        print("EXTRACTING SIGNALS FROM PROJECTS")
        print("="*80)
        print("\n8. Extracting signals...")
        records = []
        for i, project in enumerate(projects):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} projects...")
            signal = extract_signal_from_project(project)
            if signal:
                records.append(signal)
        save_extraction(
            output_dir / '08_signals.json',
            records,
            EXPECTED_COLUMNS['signals'],
            'Signals'
        )
    
    if articles:
        print("\n" + "="*80)
        print("EXTRACTING ARTICLES")
        print("="*80)
        print("\n9. Extracting articles...")
        records = []
        for i, article in enumerate(articles):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,} articles...")
            record = extract_article_record(article)
            if record:
                records.append(record)
        save_extraction(
            output_dir / '09_articles.json',
            records,
            EXPECTED_COLUMNS['articles'],
            'Articles'
        )
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\n✓ All extracted files saved to: {output_dir}")


if __name__ == "__main__":
    main()
