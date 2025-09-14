import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import time
import sys

# Command-line argument for number of compounds
parser = argparse.ArgumentParser()
parser.add_argument('--num_compounds', type=int, default=2470, help='Number of compounds to scrape')
parser.add_argument('--checkpoint_every', type=int, default=50, help='Save progress every N compounds')
args = parser.parse_args()
num_compounds = args.num_compounds
checkpoint_every = args.checkpoint_every

CSV_FILE = "ippidb_compounds.csv"
PKL_FILE = "ippidb_compounds.pkl"

# Schema for consistency
COLUMNS = [
    "compound_number", "pubchem_id", "chembl_id", "chemspider_id",
    "canonical_smiles", "iupac_name", "inchi", "inchikey",
    "biochemical_tests", "cellular_tests", "pk_tests", "cytotoxicity_tests",
    "ppi_family", "best_activity", "diseases", "mmoa",
    "error"
]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.strip()

def scrape_compound(cid, max_retries=3, backoff=5):
    url = f"https://ippidb.pasteur.fr/compounds/{cid}"
    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            pubchem_match = re.search(r"https://pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d+)", html)
            chembl_match = re.search(r"https://www\.ebi\.ac\.uk/chembldb/compound/inspect/(CHEMBL\d+)", html)
            chemspider_match = re.search(r"http://www\.chemspider\.com/Chemical-Structure\.(\d+)\.html", html)

            pubchem_id = pubchem_match.group(1) if pubchem_match else None
            chembl_id = chembl_match.group(1)[6:] if chembl_match else None
            chemspider_id = chemspider_match.group(1) if chemspider_match else None

            fields = {}
            for li in soup.find_all("li", class_="list-group-item"):
                text = li.get_text(" ", strip=True)
                if ":" in text:
                    key = text.split(":", 1)[0].strip().lower().replace(" ", "_")
                    pre = li.find("pre")
                    if pre:
                        fields[key] = pre.get_text(strip=True)

            pharma_header = soup.find("h4", string="Pharmacological data")
            biochemical = cellular = pk = cytotoxicity = None
            if pharma_header:
                table = pharma_header.find_next("table")
                if table:
                    row = table.find("tbody").find("tr")
                    if row:
                        cells = [clean_text(c.get_text(" ", strip=True)) for c in row.find_all(["th","td"])]
                        if len(cells) >= 4:
                            biochemical, cellular, pk, cytotoxicity = cells[:4]

            targets_header = soup.find("h4", string="Targets")
            ppi_family = best_activity = diseases = mmoa = None
            if targets_header:
                table = targets_header.find_next("table")
                if table:
                    row = table.find("tbody").find("tr")
                    if row:
                        cells = [clean_text(c.get_text(" ", strip=True)) for c in row.find_all("td")]
                        if len(cells) >= 4:
                            ppi_family, best_activity, diseases, mmoa = cells[:4]

            return {
                "compound_number": cid,
                "pubchem_id": pubchem_id,
                "chembl_id": chembl_id,
                "chemspider_id": chemspider_id,
                "canonical_smiles": fields.get("canonical_smiles"),
                "iupac_name": fields.get("iupac_name"),
                "inchi": fields.get("inchi"),
                "inchikey": fields.get("inchikey"),
                "biochemical_tests": biochemical,
                "cellular_tests": cellular,
                "pk_tests": pk,
                "cytotoxicity_tests": cytotoxicity,
                "ppi_family": ppi_family,
                "best_activity": best_activity,
                "diseases": diseases,
                "mmoa": mmoa,
                "error": None
            }

        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                return {"compound_number": cid, "error": str(e)}

# Load existing results safely
if os.path.exists(PKL_FILE):
    df = pd.read_pickle(PKL_FILE)
    # Ensure schema exists
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]
else:
    df = pd.DataFrame(columns=COLUMNS)

all_cids = set(range(1, num_compounds + 1))
if not df.empty:
    successful = df[df["error"].isna()]["compound_number"].tolist()
    to_scrape = list(all_cids - set(successful))
else:
    to_scrape = list(all_cids)

print(f"Resuming scraping: {len(to_scrape)} compounds left (out of {num_compounds})")

success_count = df[df["error"].isna()].shape[0]
failure_count = df[df["error"].notna()].shape[0]
batch_results = []

def flush_batch():
    global df, batch_results
    if not batch_results:
        return
    df = df[~df["compound_number"].isin([r["compound_number"] for r in batch_results])]
    df = pd.concat([df, pd.DataFrame(batch_results)], ignore_index=True)
    df.sort_values("compound_number", inplace=True)
    df.to_csv(CSV_FILE, index=False)
    df.to_pickle(PKL_FILE)
    batch_results = []

try:
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scrape_compound, cid): cid for cid in to_scrape}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(to_scrape), desc="Scraping compounds"), 1):
            result = f.result()
            batch_results.append(result)

            if result["error"]:
                failure_count += 1
                tqdm.write(f"CID {result['compound_number']} FAILED: {result['error']}")
            else:
                success_count += 1
                tqdm.write(f"CID {result['compound_number']} OK")

            if i % checkpoint_every == 0:
                flush_batch()

    flush_batch()  # final flush

except KeyboardInterrupt:
    print("\nKeyboardInterrupt received — flushing progress before exit...")
    flush_batch()
    print("Progress saved. You can rerun the script to resume.")
    sys.exit(1)

# Final retry pass
failed_df = df[df["error"].notna()]
if not failed_df.empty:
    print(f"\nRetrying {len(failed_df)} failed compounds one last time...")
    batch_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scrape_compound, cid): cid for cid in failed_df["compound_number"].tolist()}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(failed_df), desc="Final retry"), 1):
            result = f.result()
            batch_results.append(result)

            if i % checkpoint_every == 0:
                flush_batch()

    flush_batch()

success_count = df[df["error"].isna()].shape[0]
failure_count = df[df["error"].notna()].shape[0]
print(f"\nScraping complete! Success: {success_count}, Failures: {failure_count}")
print(f"Saved to {CSV_FILE} and {PKL_FILE}")

