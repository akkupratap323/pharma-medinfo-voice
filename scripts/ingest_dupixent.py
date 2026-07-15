"""Fetch the Dupixent FDA label (openFDA) and ingest it into LightRAG,
section by section, with HCP/patient audience scoping.

Usage:
  python scripts/ingest_dupixent.py --fetch          # download + write section files
  python scripts/ingest_dupixent.py --ingest hcp     # ingest full label -> HCP workspace
  python scripts/ingest_dupixent.py --ingest patient # ingest patient sections only

Scoping model: run TWO LightRAG workspaces (or instances):
  LIGHTRAG_BASE_URL_HCP     -> full prescribing information (Claire, Alex)
  LIGHTRAG_BASE_URL_PATIENT -> patient-directed sections only (Sophie)
Set both in .env; conversation config picks the base URL per persona.
"""

import argparse
import json
import os
import pathlib
import re

import httpx

# Matches numbered subsection headers like "2.1", "2.4" that open a dosing block.
_SUBSECTION_RE = re.compile(r"(?m)^\s*(\d+\.\d+)\s+")


def _split_subsections(text: str) -> list:
    """Split a section on its 'N.M' subsection headers. Returns
    [(number, text), ...]; empty if no subsections were found (caller
    then falls back to writing the section whole)."""
    marks = list(_SUBSECTION_RE.finditer(text))
    if len(marks) < 2:
        return []
    out = []
    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), text[start:end].strip()))
    return out

OPENFDA_URL = (
    'https://api.fda.gov/drug/label.json'
    '?search=openfda.brand_name.exact:"Dupixent"&limit=1'
)

OUT_DIR = pathlib.Path("data/dupixent")

# openFDA field -> (spoken section label, audience)
# audience: "hcp" = full PI (Claire/Alex), "patient" = also in Sophie's scope
SECTION_MAP = {
    "indications_and_usage":        ("Section 1, Indications and Usage", "hcp"),
    "dosage_and_administration":    ("Section 2, Dosage and Administration", "hcp"),
    "dosage_forms_and_strengths":   ("Section 3, Dosage Forms and Strengths", "hcp"),
    "contraindications":            ("Section 4, Contraindications", "hcp"),
    "warnings_and_cautions":        ("Section 5, Warnings and Precautions", "hcp"),
    "adverse_reactions":            ("Section 6, Adverse Reactions", "hcp"),
    "drug_interactions":            ("Section 7, Drug Interactions", "hcp"),
    "use_in_specific_populations":  ("Section 8, Use in Specific Populations", "hcp"),
    "clinical_pharmacology":        ("Section 12, Clinical Pharmacology", "hcp"),
    "clinical_studies":             ("Section 14, Clinical Studies", "hcp"),
    "how_supplied":                 ("Section 16, How Supplied / Storage", "hcp"),
    "information_for_patients":     ("Section 17, Patient Counseling Information", "patient"),
    "spl_patient_package_insert":   ("Patient Information", "patient"),
    "instructions_for_use":         ("Instructions for Use", "patient"),
}


def fetch() -> None:
    r = httpx.get(OPENFDA_URL, timeout=60)
    r.raise_for_status()
    result = r.json()["results"][0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Provenance: labels get revised; record exactly which version we ingested
    # so eval results stay reproducible.
    meta = {
        "brand": "DUPIXENT",
        "generic": "dupilumab",
        "spl_id": result.get("id"),
        "spl_set_id": result.get("set_id"),
        "effective_time": result.get("effective_time"),
        "version": result.get("version"),
        "source": "openFDA drug/label endpoint",
    }
    (OUT_DIR / "label_meta.json").write_text(json.dumps(meta, indent=2))

    written = 0
    for field, (label, audience) in SECTION_MAP.items():
        chunks = result.get(field)
        if not chunks:
            continue
        text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)

        # Dosing (Section 2) is one ~14K blob, so retrieval returns a digest of
        # the WHOLE section and exact per-population numbers get lost. Split on
        # its numbered subsection headers (2.1, 2.2, ...) so each dosing block —
        # adult AD, adolescent, asthma — becomes its own retrievable chunk that
        # surfaces with its exact figures intact.
        subsections = _split_subsections(text) if field == "dosage_and_administration" else None
        if subsections:
            for sub_num, sub_text in subsections:
                body = (f"DUPIXENT (dupilumab) PRESCRIBING INFORMATION\n"
                        f"{label}, subsection {sub_num}\n\n{sub_text}")
                fname = f"{audience}__{field}__{sub_num.replace('.', '_')}.txt"
                (OUT_DIR / fname).write_text(body)
                written += 1
                print(f"wrote {fname} ({len(body)} chars)")
            continue

        # Keep the section label INSIDE the text so retrieval surfaces it and
        # the agent can cite it in speech.
        body = f"DUPIXENT (dupilumab) PRESCRIBING INFORMATION\n{label}\n\n{text}"
        fname = f"{audience}__{field}.txt"
        (OUT_DIR / fname).write_text(body)
        written += 1
        print(f"wrote {fname} ({len(body)} chars)")
    print(f"\n{written} sections written. Label version: {meta['version']} "
          f"effective {meta['effective_time']}")


def ingest(scope: str) -> None:
    base_url = os.environ[
        "LIGHTRAG_BASE_URL_HCP" if scope == "hcp" else "LIGHTRAG_BASE_URL_PATIENT"
    ]
    api_key = os.environ["LIGHTRAG_API_KEY"]
    headers = {"X-API-Key": api_key}

    files = sorted(OUT_DIR.glob("*.txt"))
    if scope == "patient":
        files = [f for f in files if f.name.startswith("patient__")]
    if not files:
        raise SystemExit("no section files found — run with --fetch first")

    with httpx.Client(base_url=base_url, headers=headers, timeout=120) as cli:
        for f in files:
            r = cli.post("/documents/text", json={
                "text": f.read_text(),
                "file_source": f.name,
            })
            r.raise_for_status()
            print(f"ingested {f.name} -> {scope} workspace")
    print(f"\ndone: {len(files)} sections into {scope} scope ({base_url})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--ingest", choices=["hcp", "patient"])
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.ingest:
        ingest(a.ingest)
    if not a.fetch and not a.ingest:
        ap.print_help()
