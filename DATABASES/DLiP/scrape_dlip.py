import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
TSV_FILE = "dlip_compounds.tsv"
PKL_FILE = "dlip_compounds.pkl"
CHECKPOINT_EVERY = 50
MAX_RETRIES = 3
BACKOFF = 5
MAX_WORKERS = 10

# ---------- FIELDS ----------
COLUMNS = [
    "DLiP-ID", "DLiP-Mol-ID", "Vendor-ID",
    "Standard Inchi(RDKit)", "Standard Inchi Key(RDKit)",
    "Canonical SMILES(RDKit)", "SMILES(SDF)",
    "PPI Type(SDF)", "PDB ID(SDF)", "Receptor Chain(SDF)",
    "Protein Name Receptor(SDF)", "Peptide Chain(SDF)",
    "Protein Name Peptide(SDF)",
    "error"
]

# ---------- BASE36 HELPERS ----------
def int_to_base36(n: int) -> str:
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        n, r = divmod(n, 36)
        digits.append(chars[r])
    return "".join(reversed(digits))

def generate_ids(num=None):
    start = int("00000", 36)
    end   = int("00BQL", 36)
    count = 0
    for i in range(start, end + 1):
        if num is not None and count >= num:
            break
        yield f"D{int_to_base36(i).zfill(5)}"
        count += 1

# ---------- SCRAPER ----------
def scrape_compound(cid, max_retries=MAX_RETRIES, backoff=BACKOFF):
    url = f"https://skb-insilico.com/dlip/compound/{cid}"
    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # collect <td> label-value pairs
            data = {}
            tds = soup.find_all("td")
            for i in range(0, len(tds), 2):
                key = tds[i].get_text(strip=True)
                val = tds[i+1].get_text(strip=True) if i+1 < len(tds) else ""
                data[key] = val

            result = {col: data.get(col) for col in COLUMNS if col != "error"}
            result["error"] = None
            return result

        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                result = {col: None for col in COLUMNS[:-1]}
                result["error"] = str(e)
                return result

# ---------- STORAGE ----------
if os.path.exists(PKL_FILE):
    df = pd.read_pickle(PKL_FILE)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]
else:
    df = pd.DataFrame(columns=COLUMNS)

# ---------- ARGPARSE ----------
parser = argparse.ArgumentParser()
parser.add_argument("--num_compounds", type=int, default=None,
                    help="Number of compounds to scrape (default: all 15,214)")
args = parser.parse_args()

all_cids = list(generate_ids(num=args.num_compounds))
scraped = set(df[df["error"].isna()]["DLiP-ID"].dropna())
to_scrape = [cid for cid in all_cids if cid not in scraped]

print(f"Resuming scraping: {len(to_scrape)} left out of {len(all_cids)}")

batch_results = []

def flush_batch():
    global df, batch_results
    if not batch_results:
        return
    new_df = pd.DataFrame(batch_results)
    df = pd.concat([df[~df["DLiP-ID"].isin(new_df["DLiP-ID"])], new_df], ignore_index=True)
    df.sort_values("DLiP-ID", inplace=True)
    df.to_csv(TSV_FILE, sep="\t", index=False)   # <-- Tab-separated output
    df.to_pickle(PKL_FILE)
    batch_results = []

# ---------- MAIN LOOP ----------
try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_compound, cid): cid for cid in to_scrape}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(to_scrape), desc="Scraping compounds"), 1):
            result = f.result()
            result["DLiP-ID"] = futures[f]
            batch_results.append(result)

            if result["error"]:
                tqdm.write(f"{result['DLiP-ID']} FAILED: {result['error']}")
            else:
                tqdm.write(f"{result['DLiP-ID']} OK")

            if i % CHECKPOINT_EVERY == 0:
                flush_batch()

    flush_batch()

except KeyboardInterrupt:
    print("\nKeyboardInterrupt — saving progress...")
    flush_batch()
    sys.exit(1)

# ---------- FINAL RETRY PASS ----------
failed_df = df[df["error"].notna()]
if not failed_df.empty:
    print(f"\nRetrying {len(failed_df)} failed compounds one last time...")
    batch_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS // 2) as executor:
        futures = {executor.submit(scrape_compound, cid): cid for cid in failed_df["DLiP-ID"].tolist()}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(failed_df), desc="Final retry"), 1):
            result = f.result()
            result["DLiP-ID"] = futures[f]
            batch_results.append(result)

            if i % CHECKPOINT_EVERY == 0:
                flush_batch()

    flush_batch()

success_count = df[df["error"].isna()].shape[0]
failure_count = df[df["error"].notna()].shape[0]
print(f"\nScraping complete! Success: {success_count}, Failures: {failure_count}")
print(f"Saved to {TSV_FILE} (tab-separated) and {PKL_FILE}")

