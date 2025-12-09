#!/usr/bin/env python3
"""
Run all V4 extraction scripts against the Decidento source dump.
"""

import subprocess
import sys
from pathlib import Path


def run_extraction(script_name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print("=" * 60)

    script_path = Path(__file__).parent / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✓ {script_name} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"✗ {script_name} failed with error:")
        print(result.stderr)
        return False


def main() -> int:
    scripts = [
        "extract_01_company_basic_info.py",
        "extract_02_financial_data.py",
        "extract_03_workforce_data.py",
        "extract_04_company_structure.py",
        "extract_05_classification_flags.py",
        "extract_06_contact_metrics.py",
        "extract_07_kpi_data.py",
        "extract_08_signals.py",
        "extract_09_articles.py",
    ]

    results = {}
    for script in scripts:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            results[script] = run_extraction(script)
        else:
            print(f"⚠ {script} not found, skipping...")
            results[script] = False

    success_count = sum(1 for v in results.values() if v)
    total = len(results)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for script, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {script}")

    print(f"\nCompleted: {success_count}/{total} extractions")
    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
