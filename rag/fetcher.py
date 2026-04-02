import os
import time
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

load_dotenv()

NAME = os.getenv("NAME")
GMAIL = os.getenv("GMAIL")

SAVE_DIR = "data/raw"

COMPANIES = {
    "nvidia": "0001045810",
    "apple": "0000320193",
    "microsoft": "0000789019",
    "tesla": "0001318605",
    "amazon": "0001018724",
}

HEADERS = {
    "User-Agent": f"{NAME} {GMAIL}"
}


def get_10k_filing_index(cik: str):
    """
    Get the accession number and filing index URL for the latest 10-K.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accession_numbers = filings["accessionNumber"]

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i]
            return accession

    return None


def get_clean_text_from_index(cik: str, accession: str) -> str:
    """
    From the filing index page, find the actual 10-K document
    (not the iXBRL version) and extract clean text.
    """
    # Build the index page URL
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{accession}-index.htm"

    response = requests.get(index_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "lxml")

    # Find all document links in the filing index
    doc_link = None
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            # We want the main 10-K document, not exhibits
            if doc_type == "10-K":
                link_tag = cells[2].find("a")
                if link_tag:
                    href = link_tag["href"]
                    # Strip the XBRL viewer wrapper if present
                    if href.startswith("/ix?doc="):
                        href = href.replace("/ix?doc=", "")
                    doc_link = "https://www.sec.gov" + href
                    break

    if not doc_link:
        print("  Could not find 10-K document link in index")
        return ""

    print(f"  Document URL: {doc_link}")

    # Download and parse the actual document
    time.sleep(0.5)
    doc_response = requests.get(doc_link, headers=HEADERS)
    doc_soup = BeautifulSoup(doc_response.text, "lxml")

    # Remove script, style, and XBRL hidden tags
    for tag in doc_soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()

    # Extract clean text
    clean_text = doc_soup.get_text(separator="\n")

    # Remove excessive blank lines
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return clean_text


def download_10k(company_name: str, cik: str) -> None:
    print(f"Fetching {company_name}...")

    accession = get_10k_filing_index(cik)
    if not accession:
        print(f"  No 10-K found for {company_name}")
        return

    clean_text = get_clean_text_from_index(cik, accession)
    if not clean_text:
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{company_name}_10k.txt")

    with open(save_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(clean_text)

    print(f"  Saved to {save_path} ({len(clean_text):,} characters)")
    time.sleep(1)


def fetch_all() -> None:
    for company_name, cik in COMPANIES.items():
        download_10k(company_name, cik)
    print("\nAll done!")


if __name__ == "__main__":
    fetch_all()