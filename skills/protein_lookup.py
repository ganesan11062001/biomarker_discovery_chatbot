"""
skills/protein_lookup.py
Knowledge Layer — ProteinLookupSkill

Queries the UniProt REST API to:
  1. Convert UniProt accession IDs → gene symbols, protein names, organism
  2. Batch-annotate a list of proteins (accessions or raw names)
  3. Generate reproducible Python code for the same queries

Works offline when UniProt is unreachable (falls back to regex extraction).

Inspired by K-Dense scientific-agent-skills batch_id_converter pattern.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# UniProt REST base URL (stable v2025)
_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
_UNIPROT_ID_MAP = "https://rest.uniprot.org/idmapping"
_BATCH_SIZE = 100          # UniProt supports up to 100 IDs per request
_MAX_API_PROTEINS = 500    # cap API calls; skip to regex-only above this threshold
_REQUEST_TIMEOUT = 15      # seconds per HTTP call
_RETRY_WAIT = 2            # seconds between retries

# Regex to detect a raw UniProt accession (e.g. P12345, Q9Y6K9, A0A000)
_ACCESSION_RE = re.compile(
    r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b'
)

# Regex to pull gene name from long UniProt description "... GN=Myh6 ..."
_GN_RE = re.compile(r'\bGN=(\w[\w\-]*)', re.IGNORECASE)


def _extract_accessions(protein_strings: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Pull accession numbers out of raw protein strings.

    Returns:
        accessions  — deduplicated list of accession IDs found
        acc_to_raw  — mapping accession → original string
    """
    accessions: List[str] = []
    acc_to_raw: Dict[str, str] = {}
    seen: set = set()
    for s in protein_strings:
        for m in _ACCESSION_RE.finditer(str(s)):
            acc = m.group(1)
            if acc not in seen:
                seen.add(acc)
                accessions.append(acc)
                acc_to_raw[acc] = s
    return accessions, acc_to_raw


def _regex_gene_symbols(protein_strings: List[str]) -> List[str]:
    """Fallback: extract gene symbols from 'GN=' tags or short names."""
    symbols: List[str] = []
    seen: set = set()
    for name in protein_strings:
        name_str = str(name).strip()
        m = _GN_RE.search(name_str)
        symbol = m.group(1) if m else name_str.split(" OS=")[0].strip()
        if symbol and symbol not in seen and 1 < len(symbol) <= 30:
            if not any(c in symbol for c in "=|/\\"):
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


class ProteinLookupSkill:
    """
    Annotates proteins via the UniProt REST API.

    Usage
    -----
    skill = ProteinLookupSkill()
    result = skill.execute(
        protein_list=["P12345", "Myosin-6 OS=... GN=Myh6 ..."],
        organism="mouse",
    )
    # result["annotations"]   → list of dicts per protein
    # result["gene_symbols"]  → deduplicated list of gene symbols
    # result["analysis_code"] → re-executable Python script
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def execute(
        self,
        protein_list: List[str],
        organism: str = "human",
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        if not protein_list:
            return self._empty(output_dir)

        # Try to pull proper accessions; fall back to regex if none found
        accessions, acc_to_raw = _extract_accessions(protein_list)

        annotations: List[Dict[str, Any]] = []
        gene_symbols: List[str] = []

        if accessions:
            if len(accessions) > _MAX_API_PROTEINS:
                logger.warning(
                    "%d accessions found — truncating to %d for UniProt API to avoid timeout. "
                    "Remaining proteins will use regex gene-symbol extraction.",
                    len(accessions), _MAX_API_PROTEINS,
                )
                accessions = accessions[:_MAX_API_PROTEINS]
            annotations = self._batch_lookup(accessions)
            gene_symbols = [
                a["gene"] for a in annotations if a.get("gene")
            ]
            # De-duplicate while preserving order
            seen: set = set()
            gene_symbols = [
                g for g in gene_symbols if g not in seen and not seen.add(g)  # type: ignore[func-returns-value]
            ]

        # Fill in anything the API didn't cover with regex extraction
        api_covered = {a["accession"] for a in annotations if a.get("accession")}
        leftover = [s for s in protein_list if not any(
            acc in str(s) for acc in api_covered
        )]
        regex_symbols = _regex_gene_symbols(leftover)
        existing = set(gene_symbols)
        gene_symbols += [s for s in regex_symbols if s not in existing]

        code = self._generate_code(protein_list, organism, output_dir)

        return {
            "annotations":   annotations,
            "gene_symbols":  gene_symbols,
            "n_resolved":    len(annotations),
            "n_total":       len(protein_list),
            "analysis_code": code,
        }

    # ── UniProt batch API ─────────────────────────────────────────────────────

    def _batch_lookup(self, accessions: List[str]) -> List[Dict[str, Any]]:
        """Fetch UniProt entries in batches of _BATCH_SIZE."""
        results: List[Dict[str, Any]] = []
        for i in range(0, len(accessions), _BATCH_SIZE):
            batch = accessions[i : i + _BATCH_SIZE]
            results.extend(self._fetch_batch(batch))
            if i + _BATCH_SIZE < len(accessions):
                time.sleep(0.5)  # be polite to the API
        return results

    def _fetch_batch(self, accessions: List[str]) -> List[Dict[str, Any]]:
        ids_str = ",".join(accessions)
        url = (
            f"{_UNIPROT_BASE}/search"
            f"?query=accession:({ids_str.replace(',', '+OR+accession:')})"
            f"&fields=accession,gene_names,protein_name,organism_name,reviewed"
            f"&format=json&size={len(accessions)}"
        )
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                return [self._parse_entry(e) for e in data.get("results", [])]
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "UniProt API attempt %d failed: %s", attempt + 1, exc
                )
                if attempt < 2:
                    time.sleep(_RETRY_WAIT)
        logger.warning("UniProt API unavailable — skipping batch of %d.", len(accessions))
        return []

    @staticmethod
    def _parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        accession = entry.get("primaryAccession", "")
        # Gene name — first recommended gene name
        gene_names = entry.get("genes", [])
        gene = ""
        if gene_names:
            gn = gene_names[0]
            gene = (
                gn.get("geneName", {}).get("value", "")
                or (gn.get("synonyms") or [{}])[0].get("value", "")
            )
        # Protein name
        pn = entry.get("proteinDescription", {})
        rec = pn.get("recommendedName", {})
        protein_name = rec.get("fullName", {}).get("value", "")
        if not protein_name:
            subs = pn.get("submissionNames", [{}])
            protein_name = subs[0].get("fullName", {}).get("value", "") if subs else ""
        organism = entry.get("organism", {}).get("scientificName", "")
        reviewed = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"
        return {
            "accession":    accession,
            "gene":         gene,
            "protein_name": protein_name,
            "organism":     organism,
            "reviewed":     reviewed,
        }

    # ── Code generation ───────────────────────────────────────────────────────

    @staticmethod
    def _generate_code(
        protein_list: List[str],
        organism: str,
        output_dir: str,
    ) -> str:
        """Return a self-contained script that reproduces the UniProt lookup."""
        L: List[str] = []
        a = L.append

        a('#!/usr/bin/env python3')
        a('"""')
        a('Reproducible UniProt protein lookup / ID conversion')
        a('Auto-generated — edit PROTEIN_LIST to use your own identifiers.')
        a('"""')
        a('')
        a('import re, time, requests, pandas as pd')
        a('from pathlib import Path')
        a('')
        a('# ── Parameters ───────────────────────────────────────────────────────')
        a('PROTEIN_LIST = ' + repr(protein_list[:20]))
        a('ORGANISM     = ' + repr(organism))
        a('OUTPUT_DIR   = ' + repr(output_dir))
        a('BATCH_SIZE   = 100')
        a('')
        a('# ── Extract UniProt accessions from protein strings ──────────────────')
        a('_ACC_RE = re.compile(')
        a('    r\'\\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\\b\'')
        a(')')
        a('accessions = list(dict.fromkeys(')
        a('    m.group(1) for s in PROTEIN_LIST for m in _ACC_RE.finditer(str(s))')
        a('))')
        a('print(f"Found {len(accessions)} UniProt accessions to look up")')
        a('')
        a('# ── Batch query UniProt REST API ─────────────────────────────────────')
        a('rows = []')
        a('for i in range(0, len(accessions), BATCH_SIZE):')
        a('    batch = accessions[i : i + BATCH_SIZE]')
        a('    ids   = "+OR+accession:".join(batch)')
        a('    url   = (')
        a('        f"https://rest.uniprot.org/uniprotkb/search"')
        a('        f"?query=accession:({ids})"')
        a('        f"&fields=accession,gene_names,protein_name,organism_name,reviewed"')
        a('        f"&format=json&size={len(batch)}"')
        a('    )')
        a('    try:')
        a('        resp = requests.get(url, timeout=15)')
        a('        resp.raise_for_status()')
        a('        for e in resp.json().get("results", []):')
        a('            acc  = e.get("primaryAccession", "")')
        a('            gns  = e.get("genes", [])')
        a('            gene = gns[0].get("geneName", {}).get("value", "") if gns else ""')
        a('            pn   = e.get("proteinDescription", {})')
        a('            prot = pn.get("recommendedName", {}).get("fullName", {}).get("value", "")')
        a('            org  = e.get("organism", {}).get("scientificName", "")')
        a('            rows.append({"accession": acc, "gene": gene,')
        a('                         "protein_name": prot, "organism": org})')
        a('    except Exception as exc:')
        a('        print(f"Warning: UniProt batch failed — {exc}")')
        a('    time.sleep(0.5)')
        a('')
        a('df = pd.DataFrame(rows)')
        a('print(df.head(10).to_string(index=False))')
        a('')
        a('# ── Save ─────────────────────────────────────────────────────────────')
        a('Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)')
        a('out = str(Path(OUTPUT_DIR) / "uniprot_annotations.csv")')
        a('df.to_csv(out, index=False)')
        a('print(f"Saved → {out}")')
        a('')
        a('gene_symbols = df["gene"].dropna().tolist()')
        a('gene_symbols = list(dict.fromkeys(g for g in gene_symbols if g))')
        a('print(f"Gene symbols extracted: {gene_symbols[:20]}")')

        return '\n'.join(L)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(output_dir: str) -> Dict[str, Any]:
        return {
            "annotations":   [],
            "gene_symbols":  [],
            "n_resolved":    0,
            "n_total":       0,
            "analysis_code": "",
        }
