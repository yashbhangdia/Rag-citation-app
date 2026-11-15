"""
scripts/download_documents.py

Downloads Indian financial market documents from verified public sources.
Renames them using a consistent convention and places them in the correct
subdirectory for automatic metadata detection during ingestion.

Directory structure created:
    data/documents/
    ├── annual_reports/          ← Company annual reports
    ├── quarterly_results/       ← Quarterly earnings PDFs
    ├── regulatory/              ← SEBI circulars, RBI reports
    └── market_reports/          ← NSE/BSE market outlook reports

Naming convention (for auto-detection in document_loader.py):
    {COMPANY}_{DocType}_FY{Year}.pdf
    e.g. TCS_Annual_Report_FY2024-25.pdf

Usage:
    # Download everything
    python scripts/download_documents.py --all

    # Download only annual reports
    python scripts/download_documents.py --annual-reports

    # Download only regulatory docs
    python scripts/download_documents.py --regulatory

    # Download only quarterly results
    python scripts/download_documents.py --quarterly

    # Download a single custom URL
    python scripts/download_documents.py --url https://example.com/report.pdf --name "TCS_Annual_Report_FY2024-25.pdf" --dir annual_reports

    # List all available sources without downloading
    python scripts/download_documents.py --list
"""

import argparse
import sys
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path(__file__).parent.parent / "data" / "documents"

SUBDIRS = {
    "annual_reports":    BASE_DATA_DIR / "annual_reports",
    "quarterly_results": BASE_DATA_DIR / "quarterly_results",
    "regulatory":        BASE_DATA_DIR / "regulatory",
    "market_reports":    BASE_DATA_DIR / "market_reports",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

DOWNLOAD_TIMEOUT = 120          # seconds per file
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5            # seconds between retries
MIN_FILE_SIZE_KB = 50           # files smaller than this are likely error pages


# ── Document registry ─────────────────────────────────────────────────────────

@dataclass
class FinancialDocument:
    name:        str            # display name shown in logs
    filename:    str            # saved filename (drives metadata auto-detection)
    url:         str            # direct PDF URL
    category:    str            # maps to SUBDIRS key
    company:     str            # for grouping in --list output
    description: str            # short description


DOCUMENT_REGISTRY: list[FinancialDocument] = [

    # ── Annual Reports ────────────────────────────────────────────────────────

    FinancialDocument(
        name        = "TCS Annual Report FY2024-25",
        filename    = "TCS_Annual_Report_FY2024-25.pdf",
        url         = "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2024-25/ar/annual-report-2024-2025.pdf",
        category    = "annual_reports",
        company     = "TCS",
        description = "Integrated Annual Report covering financials, strategy, and MD&A",
    ),

    FinancialDocument(
        name        = "TCS Annual Report FY2023-24",
        filename    = "TCS_Annual_Report_FY2023-24.pdf",
        url         = "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/annual-report-2023-2024.pdf",
        category    = "annual_reports",
        company     = "TCS",
        description = "Previous year annual report for YoY comparison queries",
    ),

    FinancialDocument(
        name        = "Infosys Annual Report FY2024-25",
        filename    = "Infosys_Annual_Report_FY2024-25.pdf",
        url         = "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-25.pdf",
        category    = "annual_reports",
        company     = "Infosys",
        description = "Integrated Annual Report with financials and ESG disclosures",
    ),

    FinancialDocument(
        name        = "HDFC Bank Annual Report FY2024-25",
        filename    = "HDFC_Bank_Annual_Report_FY2024-25.pdf",
        url         = "https://www.hdfc.bank.in/content/dam/hdfcbankpws/in/en/pdf/annual-reports/2024-25/HDFC_Bank_Annual_Report_2024_25-310202.pdf",
        category    = "annual_reports",
        company     = "HDFC Bank",
        description = "Integrated Annual Report post HDFC-HDFC Bank merger",
    ),

    FinancialDocument(
        name        = "Reliance Industries Annual Report FY2024-25",
        filename    = "RIL_Annual_Report_FY2024-25.pdf",
        url         = "https://www.ril.com/reports/RIL-Integrated-Annual-Report-2024-25.pdf",
        category    = "annual_reports",
        company     = "Reliance Industries",
        description = "Integrated Annual Report covering O2C, Retail, Jio, and New Energy",
    ),

    # ── Quarterly Results ─────────────────────────────────────────────────────

    FinancialDocument(
        name        = "TCS Q4 FY2024-25 Fact Sheet",
        filename    = "TCS_Quarterly_Results_Q4FY2024-25.pdf",
        url         = "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2024-25/q4/Presentations/Q4%202024-25%20Fact%20Sheet.pdf",
        category    = "quarterly_results",
        company     = "TCS",
        description = "Q4 FY25 earnings fact sheet with revenue, margins, and headcount",
    ),

    # ── Regulatory Documents ──────────────────────────────────────────────────

    FinancialDocument(
        name        = "SEBI Master Circular Mutual Funds 2024",
        filename    = "SEBI_Circular_Mutual_Funds_FY2024.pdf",
        url         = "https://www.sebi.gov.in/sebi_data/attachdocs/jun-2024/1718367457993.pdf",
        category    = "regulatory",
        company     = "SEBI",
        description = "Master circular consolidating all mutual fund regulations",
    ),

    FinancialDocument(
        name        = "SEBI F&O Derivatives Circular Oct 2024",
        filename    = "SEBI_Circular_FnO_Derivatives_FY2024.pdf",
        url         = "https://www.sebi.gov.in/sebi_data/attachdocs/oct-2024/1728043516186.pdf",
        category    = "regulatory",
        company     = "SEBI",
        description = "Measures to strengthen equity index derivatives framework",
    ),

    FinancialDocument(
        name        = "RBI Annual Report FY2024-25",
        filename    = "RBI_Annual_Report_FY2024-25.pdf",
        url         = "https://rbidocs.rbi.org.in/rdocs/AnnualReport/PDFs/0RBI_ANNUAL_REPORT202425.pdf",
        category    = "regulatory",
        company     = "RBI",
        description = "RBI Annual Report covering monetary policy, forex, and banking supervision",
    ),

]


# ── Core downloader ───────────────────────────────────────────────────────────

def _ensure_directories() -> None:
    """Create all subdirectories if they don't exist."""
    for subdir in SUBDIRS.values():
        subdir.mkdir(parents=True, exist_ok=True)


def _file_is_valid_pdf(path: Path) -> bool:
    """Check file is a real PDF and not an HTML error page."""
    if path.stat().st_size < MIN_FILE_SIZE_KB * 1024:
        return False
    with open(path, "rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def download_file(
    url:       str,
    dest_path: Path,
    doc_name:  str,
    force:     bool = False,
) -> bool:
    """
    Download a single file with retry logic and progress bar.

    Returns True on success, False on failure.
    """
    if dest_path.exists() and not force:
        if _file_is_valid_pdf(dest_path):
            logger.info(f"  ✓ Already exists, skipping: {dest_path.name}")
            return True
        else:
            logger.warning(f"  ⚠ Existing file appears corrupt, re-downloading: {dest_path.name}")

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"  ↓ [{attempt}/{RETRY_ATTEMPTS}] Downloading: {doc_name}")
            logger.info(f"    URL: {url}")

            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=DOWNLOAD_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()

            # Check content type — reject HTML error pages
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                logger.error(
                    f"  ✗ Server returned HTML instead of PDF — URL may have changed.\n"
                    f"    Content-Type: {content_type}\n"
                    f"    Please verify URL manually: {url}"
                )
                return False

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with (
                open(dest_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"    {dest_path.name[:40]}",
                    leave=False,
                ) as bar,
            ):
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    bar.update(len(chunk))

            # Validate downloaded file
            if not _file_is_valid_pdf(dest_path):
                dest_path.unlink(missing_ok=True)
                logger.error(
                    f"  ✗ Downloaded file is not a valid PDF.\n"
                    f"    The URL may require authentication or has moved.\n"
                    f"    Please download manually from: {url}"
                )
                return False

            size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.success(f"  ✓ Saved: {dest_path.name} ({size_mb:.1f} MB)")
            return True

        except requests.exceptions.HTTPError as e:
            logger.warning(f"  ✗ HTTP {e.response.status_code} on attempt {attempt}: {doc_name}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"  ✗ Connection error on attempt {attempt}: {doc_name}")
        except requests.exceptions.Timeout:
            logger.warning(f"  ✗ Timeout on attempt {attempt}: {doc_name}")
        except Exception as e:
            logger.warning(f"  ✗ Unexpected error on attempt {attempt}: {e}")

        if attempt < RETRY_ATTEMPTS:
            logger.info(f"  ⏳ Retrying in {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)

    logger.error(
        f"  ✗ FAILED after {RETRY_ATTEMPTS} attempts: {doc_name}\n"
        f"    Manual download URL: {url}"
    )
    return False


def download_documents(
    docs:  list[FinancialDocument],
    force: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Download a list of documents.

    Returns:
        (success_count, fail_count, failed_names)
    """
    _ensure_directories()
    success, fail, failed_names = 0, 0, []

    for doc in docs:
        dest_dir  = SUBDIRS[doc.category]
        dest_path = dest_dir / doc.filename
        ok = download_file(doc.url, dest_path, doc.name, force=force)
        if ok:
            success += 1
        else:
            fail += 1
            failed_names.append(doc.name)
        time.sleep(1)   # polite delay between requests

    return success, fail, failed_names


def download_single_url(
    url:      str,
    filename: str,
    category: str,
    force:    bool = False,
) -> bool:
    """Download a single custom URL into the specified category directory."""
    _ensure_directories()
    if category not in SUBDIRS:
        logger.error(
            f"Unknown category '{category}'. "
            f"Choose from: {list(SUBDIRS.keys())}"
        )
        return False

    # Auto-add .pdf if missing
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    dest_path = SUBDIRS[category] / filename
    return download_file(url, dest_path, filename, force=force)


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_registry() -> None:
    """Print all available documents grouped by company."""
    from collections import defaultdict
    by_company: dict[str, list[FinancialDocument]] = defaultdict(list)
    for doc in DOCUMENT_REGISTRY:
        by_company[doc.company].append(doc)

    print(f"\n{'='*65}")
    print(f"  Available Documents ({len(DOCUMENT_REGISTRY)} total)")
    print(f"{'='*65}")
    for company, docs in by_company.items():
        print(f"\n  {company}")
        for doc in docs:
            dest = SUBDIRS[doc.category] / doc.filename
            status = "✓" if (dest.exists() and _file_is_valid_pdf(dest)) else "○"
            print(f"    [{status}] {doc.filename}")
            print(f"         → {doc.category}/")
            print(f"         {doc.description}")
    print(f"\n  Legend: ✓ = downloaded  ○ = not yet downloaded\n")


def print_summary(
    success:      int,
    fail:         int,
    failed_names: list[str],
) -> None:
    print(f"\n{'='*65}")
    print(f"  Download Summary")
    print(f"{'='*65}")
    print(f"  ✓ Successful : {success}")
    print(f"  ✗ Failed     : {fail}")
    if failed_names:
        print(f"\n  Failed documents (download manually):")
        for name in failed_names:
            doc = next((d for d in DOCUMENT_REGISTRY if d.name == name), None)
            if doc:
                print(f"    • {name}")
                print(f"      {doc.url}")
    print(f"\n  Files saved to: {BASE_DATA_DIR}")
    print(f"\n  Next step — ingest into vector store:")
    print(f"    python scripts/ingest.py --dir data/documents/ --reset")
    print(f"{'='*65}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Indian financial market documents for RAG ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_documents.py --all
  python scripts/download_documents.py --annual-reports
  python scripts/download_documents.py --regulatory
  python scripts/download_documents.py --quarterly
  python scripts/download_documents.py --company TCS
  python scripts/download_documents.py --list
  python scripts/download_documents.py --url https://example.com/doc.pdf \\
      --name "Wipro_Annual_Report_FY2024-25" --dir annual_reports
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",            action="store_true", help="Download all documents")
    mode.add_argument("--annual-reports", action="store_true", help="Download annual reports only")
    mode.add_argument("--quarterly",      action="store_true", help="Download quarterly results only")
    mode.add_argument("--regulatory",     action="store_true", help="Download regulatory documents only")
    mode.add_argument("--market-reports", action="store_true", help="Download market reports only")
    mode.add_argument("--company",        type=str,            help="Download all docs for a specific company e.g. 'TCS'")
    mode.add_argument("--url",            type=str,            help="Download a single custom URL")
    mode.add_argument("--list",           action="store_true", help="List all available sources")

    parser.add_argument("--name", type=str, help="Filename for --url download (without .pdf)")
    parser.add_argument("--dir",  type=str, help="Category dir for --url: annual_reports | quarterly_results | regulatory | market_reports")
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        print_registry()
        return

    # ── Single URL mode ───────────────────────────────────────────────────────
    if args.url:
        if not args.name or not args.dir:
            print("--url requires both --name and --dir. Use --help for examples.")
            sys.exit(1)
        ok = download_single_url(args.url, args.name, args.dir, force=args.force)
        sys.exit(0 if ok else 1)

    # ── Registry-based download ───────────────────────────────────────────────
    if args.all:
        docs = DOCUMENT_REGISTRY
        label = "all documents"

    elif args.annual_reports:
        docs = [d for d in DOCUMENT_REGISTRY if d.category == "annual_reports"]
        label = "annual reports"

    elif args.quarterly:
        docs = [d for d in DOCUMENT_REGISTRY if d.category == "quarterly_results"]
        label = "quarterly results"

    elif args.regulatory:
        docs = [d for d in DOCUMENT_REGISTRY if d.category == "regulatory"]
        label = "regulatory documents"

    elif args.market_reports:
        docs = [d for d in DOCUMENT_REGISTRY if d.category == "market_reports"]
        label = "market reports"

    elif args.company:
        docs = [
            d for d in DOCUMENT_REGISTRY
            if d.company.lower() == args.company.lower()
        ]
        if not docs:
            available = sorted({d.company for d in DOCUMENT_REGISTRY})
            print(f"No documents found for company '{args.company}'.")
            print(f"Available companies: {', '.join(available)}")
            sys.exit(1)
        label = f"{args.company} documents"

    else:
        docs = []
        label = ""

    if not docs:
        print(f"No documents matched. Use --list to see available sources.")
        sys.exit(0)

    print(f"\n{'='*65}")
    print(f"  Downloading {len(docs)} {label}")
    print(f"{'='*65}\n")

    success, fail, failed_names = download_documents(docs, force=args.force)
    print_summary(success, fail, failed_names)

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()




'''
# See all available documents and download status
python scripts/download_documents.py --list

# Download everything at once
python scripts/download_documents.py --all

# Download by category
python scripts/download_documents.py --annual-reports
python scripts/download_documents.py --regulatory
python scripts/download_documents.py --quarterly

# Download all docs for one company
python scripts/download_documents.py --company TCS
python scripts/download_documents.py --company "HDFC Bank"

# Download a custom URL you found manually
python scripts/download_documents.py \
    --url https://www.wipro.com/content/dam/wipro/annual-report.pdf \
    --name "Wipro_Annual_Report_FY2024-25" \
    --dir annual_reports

# Force re-download even if file exists
python scripts/download_documents.py --all --force
'''