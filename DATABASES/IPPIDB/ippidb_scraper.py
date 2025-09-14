import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Command-line argument for number of compounds
parser = argparse.ArgumentParser()
parser.add_argument('--num_compounds', type=int, default=2470, help='Number of compounds to scrape')
args = parser.parse_args()
num_compounds = args.num_compounds

# Helper to clean whitespace and commas
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.strip()

# Function to scrape a single compound
def scrape_compound(cid):
    url = f"https://ippidb.pasteur.fr/compounds/{cid}"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        
        # Regex-based IDs
        pubchem_match = re.search(r"https://pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d+)", html)
        chembl_match = re.search(r"https://www\.ebi\.ac\.uk/chembldb/compound/inspect/(CHEMBL\d+)", html)
        chemspider_match = re.search(r"http://www\.chemspider\.com/Chemical-Structure\.(\d+)\.html", html)

        pubchem_id = pubchem_match.group(1) if pubchem_match else None
        chembl_id = chembl_match.group(1)[6:] if chembl_match else None  # numeric only
        chemspider_id = chemspider_match.group(1) if chemspider_match else None

        # Dynamic chemical fields
        fields = {}
        for li in soup.find_all("li", class_="list-group-item"):
            text = li.get_text(" ", strip=True)
            if ":" in text:
                key = text.split(":", 1)[0].strip().lower().replace(" ", "_")
                pre = li.find("pre")
                if pre:
                    fields[key] = pre.get_text(strip=True)

        # Pharmacological table
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

        # Targets table
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
            "mmoa": mmoa
        }

    except Exception as e:
        print(f"Failed to scrape compound {cid}: {e}")
        return {"compound_number": cid}

# Scrape all compounds in parallel
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(scrape_compound, cid): cid for cid in range(1, num_compounds+1)}
    for f in tqdm(as_completed(futures), total=num_compounds, desc="Scraping compounds"):
        results.append(f.result())

# Create DataFrame
df = pd.DataFrame(results)
df.sort_values("compound_number", inplace=True)

# Save CSV and Pickle
df.to_csv("ippidb_compounds.csv", index=False)
df.to_pickle("ippidb_compounds.pkl")

print("Scraping complete! Saved to ippidb_compounds.csv and ippidb_compounds.pkl")

