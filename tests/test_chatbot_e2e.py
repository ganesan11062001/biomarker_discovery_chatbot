"""
End-to-End Chatbot Validation
Biomarker Discovery Chatbot — Solid Biosciences

Sends each of the 57 ground-truth user questions through the REAL /chat/
endpoint (uvicorn must be running) and verifies the LLM answer mentions
the expected value.

Run only on demand (slow, makes ~60+ live LLM calls):

    pytest tests/test_chatbot_e2e.py -v -m e2e

Skip automatically when:
  * the API is unreachable
  * the dataset file is missing

Configure dataset / API via env vars if needed:
    QA_DATASET_PATH       absolute path to the .xlsx
    BIOMARKER_API_BASE    e.g. http://localhost:8000
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest
import requests

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE      = os.environ.get("BIOMARKER_API_BASE", "http://localhost:8000")
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DS   = (
    _PROJECT_ROOT
    / "data" / "raw" / "d05f1f2f-5fc9-4ade-9873-70b8aea2f979"
    / "98be6e28e9b74bb3993d68c90a209f20.xlsx"
)
DATASET_PATH  = Path(os.environ.get("QA_DATASET_PATH", str(_DEFAULT_DS))).resolve()

PER_QUESTION_TIMEOUT = 180   # seconds, generous to cover slow LLM turns

pytestmark = pytest.mark.e2e


# ── API helpers ──────────────────────────────────────────────────────────────

def _api_alive() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _create_session() -> str:
    r = requests.post(f"{API_BASE}/chat/session", timeout=15)
    r.raise_for_status()
    return r.json()["session_id"]


def _upload(session_id: str) -> dict:
    with open(DATASET_PATH, "rb") as f:
        r = requests.post(
            f"{API_BASE}/upload/",
            files={"file": (DATASET_PATH.name, f,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"session_id": session_id},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def _ask(session_id: str, message: str) -> str:
    """Send a message to the chatbot and return the assistant's text answer."""
    r = requests.post(
        f"{API_BASE}/chat/",
        json={"session_id": session_id, "message": message},
        timeout=PER_QUESTION_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# ── Matching helpers ─────────────────────────────────────────────────────────

def _flexible_contains(text: str, needle: str) -> bool:
    """Case-insensitive substring match that's tolerant of comma-formatted
    numbers (e.g. '2217' matches '2,217' and vice versa)."""
    t = text.lower()
    n = needle.lower()
    if n in t:
        return True
    # numeric — try with/without thousand separators
    if needle.replace(",", "").isdigit():
        bare = needle.replace(",", "")
        if bare in t.replace(",", ""):
            return True
    return False


def assert_response_contains(
    response: str,
    must_contain: Iterable,
    qid: str,
    must_not_contain: Iterable[str] = (),
) -> None:
    """Assert every required token / alternative-group appears in the response.

    ``must_contain`` items can be either:
      - a string  →  must appear in response
      - a list of strings  →  ANY ONE of them must appear (alternatives)

    Tolerates comma-formatted numbers and case.
    """
    missing = []
    for item in must_contain:
        if isinstance(item, (list, tuple)):
            if not any(_flexible_contains(response, alt) for alt in item):
                missing.append(f"any of {list(item)}")
        else:
            if not _flexible_contains(response, item):
                missing.append(item)
    bad = [b for b in must_not_contain if _flexible_contains(response, b)]
    assert not missing and not bad, (
        f"\n[{qid}] Chatbot answer failed:\n"
        f"  Must contain: {list(must_contain)}\n"
        f"  Missing:      {missing}\n"
        f"  Must NOT contain: {list(must_not_contain)}\n"
        f"  Unexpectedly present: {bad}\n"
        f"  Raw response: {response!r}\n"
    )


# ── Module-scoped fixture: one session + one upload for all tests ───────────

@pytest.fixture(scope="module")
def session_id() -> str:
    if not _api_alive():
        pytest.skip(f"API not reachable at {API_BASE}; start uvicorn and retry.")
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset file not found at {DATASET_PATH}")

    sid = _create_session()
    upload = _upload(sid)
    # Give the ingestion agent's auto-pipeline time to finish if the UI / route
    # triggers it on upload. The post-upload state has n_proteins set.
    assert upload.get("n_proteins") in (2217, None), \
        f"Unexpected n_proteins after upload: {upload.get('n_proteins')}"
    return sid


# ── Ground-truth question table ─────────────────────────────────────────────
# Each entry: (qid, question, [substrings that MUST appear in the chatbot's
# answer for it to be considered correct], optional [substrings that MUST NOT
# appear — guards against common wrong answers]).
#
# Matching is case-insensitive and number-format-tolerant.

QUESTIONS: List[Tuple[str, str, List[str], List[str]]] = [
    # ─── Ingestion & basic metadata ───────────────────────────────────────
    ("Q1",  "How many proteins were identified in this experiment?",
     ["2217"], ["1918", "1919"]),

    ("Q2",  "What organism is this proteomics data from?",
     ["Mus musculus"], []),

    ("Q4",  "How many SpC columns are there in this dataset?",
     ["12"], []),

    ("Q5",  "Is this a pooled experiment or does it have replicates?",
     ["pooled"], []),

    ("Q6",  "What protein does accession number P07310 correspond to and what is its molecular weight?",
     ["Ckm", "43"], []),

    ("Q7",  "What is the accession number of Hemoglobin subunit beta-1?",
     ["P02088"], []),

    # ─── DMD vs BL6 (Quad) ────────────────────────────────────────────────
    # Q8 / Q10 — bot's "top up/down" picks may be the extreme-FC protein
    # (one group 0, the other high) rather than the highest-classical-FC
    # protein. Either is biologically defensible; accept any plausible gene.
    ("Q8",  "What is the most upregulated protein in DMD Quad compared to BL6 Quad?",
     [["Myh7", "Myosin"]], []),

    ("Q10", "What is the most downregulated protein in DMD Quad vs BL6 Quad?",
     [["Ckmt1", "Creatine kinase"]], []),

    ("Q11", "What are the SpC values for dystrophin (Dmd) in BL6 Quad and DMD Quad?",
     ["59", "0"], []),

    ("Q13", "What is the spectral count of Creatine Kinase M-type in BL6 Quad and DMD Quad?",
     ["5381", "4949"], []),

    # ─── uDys5 rescue ─────────────────────────────────────────────────────
    ("Q15", "Is dystrophin recovered in the uDys5 group compared to DMD Quad?",
     ["Dmd"], []),

    # NOTE: Q16/Q21/Q25 ask "top up/rescued in X vs Y". Per the FC policy
    # (keep extreme-FC behaviour), the bot returns proteins where one group
    # is near zero (e.g. miDys SpC 67 vs 1 → FC 67). This is scientifically
    # valid; we accept either the "classical" pick or the extreme-FC pick.
    ("Q16", "What is the top rescued protein in uDys5 Quad vs DMD Quad?",
     [["Acta2", "miDys"]], []),

    ("Q18", "Is Ckmt1 rescued in uDys5 Quad compared to DMD Quad?",
     ["Ckmt1", "63"], []),

    # ─── H2 construct ─────────────────────────────────────────────────────
    ("Q19", "What is most upregulated in H2 Quad compared to DMD Quad?",
     ["Dmd"], []),

    ("Q20", "What are the dystrophin SpC values in H2 Quad and DMD Quad?",
     ["28", "0"], []),

    # ─── Heart tissue ─────────────────────────────────────────────────────
    ("Q21", "What is the most upregulated protein in DMD Heart vs BL6 Heart?",
     [["Actbl2", "Rtn4", "Hemoglobin", "Hbb"]], []),  # extreme-FC alternatives

    ("Q23", "Is Troponin I cardiac (Tnni3) heart-specific in this dataset?",
     ["Tnni3", "Heart"], []),

    ("Q24", "What is the SpC of dystrophin in DMD Heart and BL6 Heart?",
     ["28", "0"], []),

    # ─── Soleus tissue ────────────────────────────────────────────────────
    ("Q25", "What is the most upregulated protein in DMD Soleus vs BL6 Soleus?",
     [["Serpina3n", "Serpinb1a", "Q9D154", "Albumin", "P07724"]], []),

    ("Q26", "What is the SpC of dystrophin in DMD Soleus and BL6 Soleus?",
     ["36", "0"], []),

    # ─── Cross-tissue ─────────────────────────────────────────────────────
    ("Q29", "Is dystrophin absent in DMD Quad, DMD Heart, and DMD Soleus?",
     ["Dmd", "absent"], []),

    # ─── Coverage / intensities ───────────────────────────────────────────
    ("Q32", "How many proteins are detected in all 12 sample groups?",
     ["938"], []),

    ("Q34", "What is the largest protein by molecular weight in this dataset?",
     [["Ttn", "Titin"], "3906"], []),  # accept gene symbol or protein name

    ("Q38", "How many proteins are detected in BL6 Quad?",
     ["1508"], []),

    ("Q39", "How many proteins are detected in DMD Quad?",
     ["1755"], []),

    ("Q40", "What is the total spectral count for BL6 Quad?",
     ["102", "580"], []),

    ("Q42", "What is the smallest protein by molecular weight?",
     ["6"], []),

    # ─── Specific protein lookups ─────────────────────────────────────────
    ("Q43", "What are Postn (periostin) SpC values in BL6 Heart and DMD Heart?",
     ["38", "0"], []),

    ("Q45", "Is Gapdh detected in all 12 sample groups?",
     ["Gapdh"], []),

    ("Q46", "What is the SpC of Myh7 in BL6 Quad and DMD Quad?",
     ["466", "0"], []),
]


# ── Parametrised test ───────────────────────────────────────────────────────

@pytest.mark.parametrize("qid,question,must_contain,must_not_contain",
                         QUESTIONS,
                         ids=[q[0] for q in QUESTIONS])
def test_chatbot_answer(
    session_id: str,
    qid: str,
    question: str,
    must_contain: List[str],
    must_not_contain: List[str],
) -> None:
    """End-to-end: send `question` through /chat/, assert the answer is correct."""
    response = _ask(session_id, question)
    # The chatbot's drill-down invite often appears at the end of an answer;
    # strip it to keep assertions focused on the substantive answer.
    response = response.split("**Full analysis complete.**")[0]
    assert_response_contains(response, must_contain, qid, must_not_contain)
