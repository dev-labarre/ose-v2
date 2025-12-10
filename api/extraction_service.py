"""Service for extracting data from source format JSON into 9 DataFrames."""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import extraction functions from extract_all.py
from src.extract_all import (
    extract_company_basic_info,
    extract_financial_data,
    extract_workforce_data,
    extract_company_structure,
    extract_classification_flags,
    extract_contact_metrics,
    extract_kpi_data,
    extract_signal_from_project,
    extract_article_record,
)


class ExtractionService:
    """Service for extracting 9 datasets from source format JSON."""
    
    def extract_from_source(self, companies: Optional[List[dict]] = None,
                          articles: Optional[List[dict]] = None,
                          projects: Optional[List[dict]] = None) -> Dict[str, List[dict]]:
        """
        Extract 9 datasets from source format JSON.
        
        Args:
            companies: List of company objects (from agro_alim_companies.json format)
            articles: List of article objects (from agro_alim_articles.json format)
            projects: List of project objects (from agro_alim_projects.json format)
        
        Returns:
            Dictionary with 9 datasets ready for merge_on_siren()
        """
        data_dict = {
            "company_basic": [],
            "financial": [],
            "workforce": [],
            "structure": [],
            "flags": [],
            "contact": [],
            "kpi": [],
            "signals": [],
            "articles": []
        }
        
        # Extract from companies
        if companies:
            for company in companies:
                # Extract basic info
                data_dict["company_basic"].append(extract_company_basic_info(company))
                # Extract financial
                data_dict["financial"].append(extract_financial_data(company))
                # Extract workforce
                data_dict["workforce"].append(extract_workforce_data(company))
                # Extract structure
                data_dict["structure"].append(extract_company_structure(company))
                # Extract flags
                data_dict["flags"].append(extract_classification_flags(company))
                # Extract contact
                data_dict["contact"].append(extract_contact_metrics(company))
                # Extract KPI (can have multiple records per company)
                kpi_records = extract_kpi_data(company)
                data_dict["kpi"].extend(kpi_records)
        
        # Extract signals from projects
        if projects:
            for project in projects:
                signal = extract_signal_from_project(project)
                if signal:
                    data_dict["signals"].append(signal)
        
        # Extract articles
        if articles:
            for article in articles:
                record = extract_article_record(article)
                if record:
                    data_dict["articles"].append(record)
        
        return data_dict

