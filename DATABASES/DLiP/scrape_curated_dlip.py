import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import sys
import argparse
from tqdm import tqdm

# ---------- CONFIG ----------
TSV_FILE = "compounds.tsv"
PKL_FILE = "compounds.pkl"
CHECKPOINT_EVERY = 50
MAX_RETRIES = 3
BACKOFF = 5
MAX_CONSECUTIVE_FAILS = 20  # stop after N fails in a row

# ---------- PREFIX MAP ----------
PREFIX_MAP = {
    "I": "iPPI-DB",
    "P": "2P2I-DB",
    "T": "TIMBAL",
    "C": "ChEMBL",
    "J": "Journal_of_Medical_Chemistry",
}
prefixes = list(PREFIX_MAP.keys())

# ---------- HEX HELPERS ----------
def int_to_hex(n: int) -> str:
    return format(n, "05X")  # 5-digit zero-padded hex

# ---------- FIELDS ----------
FIELDS = [
    "DLiP-ID","DLiP-Mol-ID","Vendor-ID",
    "Standard Inchi(RDKit)","Standard Inchi Key(RDKit)","Canonical SMILES(RDKit)",
    "SMILES(SDF)","MW(SDF)","MW(RDKit)","MW Monoisotopic(RDKit)","MolLogP(RDKit)",
    "XLogP(SDF)","XLogP(CDK)","ALogP(SDF)","Num H Acceptors(SDF)","nHAcceptors(SDF)",
    "HBA(RDKit)","HBA Lipinski(RDKit)","Num H Donors(SDF)","nHDonors(SDF)",
    "HBD(RDKit)","HBD Lipinski(RDKit)","PSA(SDF)","PSA(RDKit)","RO3 Pass",
    "Num RO5 Violations","Num Lipinski RO5 Violations","nBonds(SDF)","nRigidBonds(SDF)",
    "nRotatableBonds(SDF)","nRotatableBonds(RDKit)","nRings(SDF)","nRings(RDKit)",
    "Aromatic Rings(RDKit)","nAtoms(SDF)","Heavy Atoms(RDKit)","QED Weighted(RDKit)",
    "Molecular Formula(SDF)","PPI Type(SDF)","PDB ID(SDF)","Receptor Chain(SDF)",
    "Protein Name Receptor(SDF)","Peptide Chain(SDF)","Protein Name Peptide(SDF)",
    "ELM ID(SDF)","Motif Sequence(SDF)","Physical Form Apperance(SDF)",
    "Quantity of Sample mg(SDF)","Box Num(SDF)","Box Position(SDF)","fCsp3(SDF)",
    "nCarbons(SDF)","nHetAtoms(SDF)","nHalide(SDF)","Year(SDF)","error"
]

# ---------- SCRAPER ----------
def scrape_compound(cid, max_retries=MAX_RETRIES, backoff=BACKOFF):
    url = f"https://skb-insilico.com/dlip/compound/{cid}"
    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            data = {field: None for field in FIELDS}
            data["DLiP-ID"] = cid

            rows = soup.find_all("tr")
            for tr in rows:
                tds = tr.find_all("td")
                if len(tds) != 2:
                    continue
                key = tds[0].get_text(strip=True)
                if key not in FIELDS:
                    continue
                if key == "PDB ID(SDF)":
                    a = tds[1].find("a")
                    data[key] = a.get_text(strip=True) if a else tds[1].get_text(strip=True)
                else:
                    data[key] = tds[1].get_text(strip=True)

            data["error"] = None
            return data

        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                return {"DLiP-ID": cid, "error": str(e)}

# ---------- STORAGE ----------
if os.path.exists(PKL_FILE):
    df = pd.read_pickle(PKL_FILE)
else:
    df = pd.DataFrame(columns=FIELDS)

batch_results = []

def save_split_files(df):
    """Save a CSV and PKL for each database prefix."""
    for prefix, db_name in PREFIX_MAP.items():
        sub = df[df["DLiP-ID"].str.startswith(prefix)]
        if sub.empty:
            continue
        sub = sub.sort_values(by="DLiP-ID").reset_index(drop=True)
        sub.to_csv(f"{db_name}_compounds.csv", sep="\t", index=False)
        sub.to_pickle(f"{db_name}_compounds.pkl")

def flush_batch():
    global df, batch_results
    if not batch_results:
        return
    new_df = pd.DataFrame(batch_results)
    new_cols = [col for col in new_df.columns if col not in df.columns]
    if new_cols:
        df = pd.concat([df, pd.DataFrame(columns=new_cols)], ignore_index=True)
    missing_in_new = [col for col in df.columns if col not in new_df.columns]
    if missing_in_new:
        new_df = pd.concat([new_df, pd.DataFrame(columns=missing_in_new)], ignore_index=True)
    df = pd.concat([df[~df["DLiP-ID"].isin(new_df["DLiP-ID"])], new_df], ignore_index=True)

    # Sort and save
    df = df.sort_values(by="DLiP-ID").reset_index(drop=True)
    df.to_csv(TSV_FILE, sep="\t", index=False)
    df.to_pickle(PKL_FILE)
    save_split_files(df)

    batch_results.clear()

# ---------- ARGPARSE ----------
parser = argparse.ArgumentParser()
parser.add_argument("--num_compounds", type=int, default=None,
                    help="Number of compounds to scrape per prefix (default: unlimited until failures)")
args = parser.parse_args()

# ---------- MAIN LOOP ----------
try:
    for prefix in prefixes:
        print(f"\n=== Starting prefix {prefix} ({PREFIX_MAP[prefix]}) ===")
        i = 0
        consecutive_fails = 0

        while True:
            cid = f"{prefix}{int_to_hex(i)}"
            i += 1

            if args.num_compounds and i > args.num_compounds:
                break
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print(f"Stopping {prefix} after {consecutive_fails} consecutive failures.")
                break

            result = scrape_compound(cid)
            result["DLiP-ID"] = cid
            batch_results.append(result)

            if result["error"]:
                tqdm.write(f"{result['DLiP-ID']} FAILED: {result['error']}")
                consecutive_fails += 1
            else:
                tqdm.write(f"{result['DLiP-ID']} OK")
                consecutive_fails = 0  # reset on success

            if len(batch_results) % CHECKPOINT_EVERY == 0:
                flush_batch()

        flush_batch()

except KeyboardInterrupt:
    print("\nKeyboardInterrupt — saving progress...")
    flush_batch()
    sys.exit(1)

# ---------- FINAL SAVE ----------
df = df.sort_values(by="DLiP-ID").reset_index(drop=True)
df.to_csv(TSV_FILE, sep="\t", index=False)
df.to_pickle(PKL_FILE)
save_split_files(df)

success_count = df[df["error"].isna()].shape[0]
failure_count = df[df["error"].notna()].shape[0]
print(f"\nScraping complete! Success: {success_count}, Failures: {failure_count}")
print("Saved combined files and per-database CSV/PKL files.")

