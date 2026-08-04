"""
scripts/run_scenario_tests.py

Drives all 10 synthetic analysis scenarios through the REAL backend API
(upload -> chat -> results), end to end, exactly as the frontend would.
Verifies recovered results against the known ground-truth signal baked
into each synthetic file by generate_synthetic_scenarios.py, and writes:

  logs/scenario_test_report_<timestamp>.md     — human-readable report
  logs/scenario_test_transcript_<timestamp>.jsonl — full raw request/response log

Requires the FastAPI backend running locally (make api / uvicorn api.main:app).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synthetic_scenarios"
LOG_DIR = ROOT / "logs"
TIMEOUT = 180


def _spiked_ids(prefix: str, indices: List[int]) -> List[str]:
    return [f"{prefix}-{i + 1}" for i in indices]


SCENARIOS: List[Dict[str, Any]] = [
    {
        "num": 1, "name": "Single pooled plasma sample",
        "file": "01_pooled_plasma_single_sample.xlsx",
        "questions": ["Which proteins are unusually abundant in this sample?"],
        "spiked": _spiked_ids("OlinkProtein", list(range(10))),
        "check": "descriptive_top",
    },
    {
        "num": 2, "name": "Small case-control n=3v3",
        "file": "02_case_control_n3v3.xlsx",
        "questions": ["Which proteins differ between Healthy and Disease?"],
        "spiked": _spiked_ids("InflammationPanel", list(range(8))),
        "check": "two_group_up",
    },
    {
        "num": 3, "name": "Paired pre/post treatment",
        "file": "03_paired_pre_post.xlsx",
        "questions": [
            "Run a paired analysis comparing Baseline vs PostTreatment — which proteins changed after treatment?"
        ],
        "spiked": _spiked_ids("InflammationPanel", list(range(10))),
        "check": "two_group_down",
    },
    {
        "num": 4, "name": "Discovery MS n=5v5",
        "file": "04_discovery_ms_n5v5.xlsx",
        "questions": ["What are the top differentially expressed proteins between Control and Disease?"],
        "spiked": _spiked_ids("TumorProtein", list(range(15))),
        "check": "two_group_up",
    },
    {
        "num": 5, "name": "TMT multiplex 4 groups",
        "file": "05_tmt_multiplex_4groups.xlsx",
        "questions": [
            "Run a one-way ANOVA across Vehicle, TreatmentA, TreatmentB, and TreatmentC — which proteins differ across treatments, and which pairs of groups differ?"
        ],
        "spiked": _spiked_ids("CellLineProtein", list(range(15))),
        "check": "anova_tukey",
    },
    {
        "num": 6, "name": "Dose-response 4 levels",
        "file": "06_dose_response_4levels.xlsx",
        "questions": [
            "Run a dose-response trend test across Dose0_Vehicle=0, Dose1_Low=1, Dose2_Medium=2, Dose3_High=3 — which proteins show a dose-dependent response?"
        ],
        "spiked": _spiked_ids("CellCultureProtein", list(range(12))),
        "check": "dose_response",
    },
    {
        "num": 7, "name": "Time-course 5 timepoints",
        "file": "07_time_course_5timepoints.xlsx",
        "questions": [
            "Run the repeated-measures analysis now across Day0, Day1, Day3, Day7, Day14, treating samples ending in the same Subj number (Subj1-Subj4) as the same subject across timepoints. Tell me which proteins change significantly over time."
        ],
        "spiked": _spiked_ids("PlasmaProtein", list(range(10))),
        "check": "repeated_measures",
    },
    {
        "num": 8, "name": "Disease-severity subtypes",
        "file": "08_disease_severity_subtypes.xlsx",
        "questions": [
            "Run a one-way ANOVA across Healthy, Mild, Moderate, and Severe — which biomarkers distinguish disease severity?"
        ],
        "spiked": _spiked_ids("SerumProtein", list(range(12))),
        "check": "anova_tukey",
    },
    {
        "num": 9, "name": "Phosphoproteomics/PTM",
        "file": "09_phosphoproteomics_ptm.xlsx",
        "questions": [
            "Compare Tumor vs Normal — which phosphosites are significantly altered, and which kinases or pathways are implicated?"
        ],
        "spiked": None,
        "check": "ptm",
    },
    {
        "num": 10, "name": "Large clinical cohort",
        "file": "10_large_clinical_cohort.xlsx",
        "questions": [
            "Run the differential expression analysis now comparing Responder vs NonResponder — which proteins differ significantly?",
            "Now run a logistic regression predicting responder status (Responder=1, NonResponder=0) from protein levels and report the ROC AUC.",
        ],
        "spiked": _spiked_ids("PlasmaBiomarker", list(range(20))),
        "check": "clinical_cohort",
    },
]


def upload(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{BASE}/upload/",
            files={"file": (file_path.name, f,
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            timeout=TIMEOUT,
        )
    return {"status_code": r.status_code, "body": _safe_json(r)}


def chat(session_id: str, message: str) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/chat/",
        json={"session_id": session_id, "message": message},
        timeout=TIMEOUT,
    )
    return {"status_code": r.status_code, "body": _safe_json(r)}


def get_results(session_id: str) -> Dict[str, Any]:
    r = requests.get(f"{BASE}/results/{session_id}", timeout=TIMEOUT)
    return {"status_code": r.status_code, "body": _safe_json(r)}


def _safe_json(r: requests.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text[:2000]}


def _protein_field(bm: Dict[str, Any]) -> str:
    return str(bm.get("protein") or bm.get("accession") or "")


def verify(scenario: Dict[str, Any], results_body: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic ground-truth check against the spiked signal. Returns
    {"verdict": PASS/PARTIAL/FAIL, "notes": [...]}."""
    notes: List[str] = []
    error_message = results_body.get("error_message")
    status_ = results_body.get("status")
    top = results_body.get("top_biomarkers") or []

    if error_message:
        return {"verdict": "FAIL", "notes": [f"error_message: {error_message}"]}
    if status_ == "error":
        return {"verdict": "FAIL", "notes": [f"status=error"]}
    if not top:
        return {"verdict": "FAIL", "notes": ["top_biomarkers is empty"]}

    check = scenario["check"]
    spiked = scenario.get("spiked") or []
    top_ids = [_protein_field(b) for b in top[:max(len(spiked), 10) + 5]]

    def _hit_count() -> int:
        return sum(1 for sid in spiked if any(sid in tid for tid in top_ids))

    if check in ("two_group_up", "two_group_down", "descriptive_top", "anova_tukey", "dose_response", "repeated_measures"):
        hits = _hit_count()
        frac = hits / max(len(spiked), 1)
        notes.append(f"{hits}/{len(spiked)} spiked proteins recovered in top results")
        if frac >= 0.5:
            verdict = "PASS"
        elif frac > 0:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        if check == "two_group_down":
            neg = [b for b in top if any(sid in _protein_field(b) for sid in spiked)
                   and (b.get("log2_fold_change") or 0) < 0]
            notes.append(f"{len(neg)} recovered spiked proteins have the expected negative log2FC (post < baseline)")
            if not neg and hits > 0:
                verdict = "PARTIAL"

        if check == "anova_tukey":
            tukey_col_present = any("tukey_significant_pairs" in b for b in top)
            notes.append(f"tukey_significant_pairs column present: {tukey_col_present}")
            if not tukey_col_present:
                verdict = "PARTIAL" if verdict == "PASS" else verdict

        if check == "dose_response":
            slope_col_present = any("dose_slope" in b for b in top)
            notes.append(f"dose_slope column present: {slope_col_present}")
            if not slope_col_present:
                verdict = "PARTIAL" if verdict == "PASS" else verdict

        if check == "repeated_measures":
            # This path depends on the LLM correctly building subject_map from
            # natural language — record what actually happened either way.
            notes.append(f"analysis_mode/status reflect whatever the LLM routed to (see raw response)")

        return {"verdict": verdict, "notes": notes}

    if check == "ptm":
        pathways = results_body.get("pathways") or []
        notes.append(f"top_biomarkers non-empty: {bool(top)}; pathways returned: {len(pathways)}")
        verdict = "PASS" if top and pathways else ("PARTIAL" if top else "FAIL")
        return {"verdict": verdict, "notes": notes}

    if check == "clinical_cohort":
        hits = _hit_count()
        notes.append(f"{hits}/{len(spiked)} spiked biomarkers recovered in top results (two-group pass)")
        verdict = "PASS" if hits / max(len(spiked), 1) >= 0.5 else ("PARTIAL" if hits > 0 else "FAIL")
        return {"verdict": verdict, "notes": notes}

    return {"verdict": "PARTIAL", "notes": ["no verification rule for this check type"]}


def run_scenario(scenario: Dict[str, Any], transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    file_path = DATA_DIR / scenario["file"]
    record: Dict[str, Any] = {"num": scenario["num"], "name": scenario["name"], "file": scenario["file"]}

    t0 = time.time()
    up = upload(file_path)
    transcript.append({"scenario": scenario["num"], "step": "upload", **up})
    record["upload_status"] = up["status_code"]
    if up["status_code"] not in (200, 201):
        record["verdict"] = "FAIL"
        record["notes"] = [f"upload failed: {up['body']}"]
        return record

    session_id = up["body"]["session_id"]
    record["session_id"] = session_id
    record["n_proteins"] = up["body"].get("n_proteins")
    record["n_samples"] = up["body"].get("n_samples")
    record["data_type"] = up["body"].get("data_type")

    chat_responses = []
    for q in scenario["questions"]:
        c = chat(session_id, q)
        transcript.append({"scenario": scenario["num"], "step": "chat", "question": q, **c})
        chat_responses.append({
            "question": q,
            "status_code": c["status_code"],
            "intent": (c["body"] or {}).get("intent"),
            "status": (c["body"] or {}).get("status"),
            "response_excerpt": ((c["body"] or {}).get("response") or "")[:600],
        })
    record["chat_responses"] = chat_responses

    res = get_results(session_id)
    transcript.append({"scenario": scenario["num"], "step": "results", **res})
    verdict_info = verify(scenario, res["body"] or {})
    record["verdict"] = verdict_info["verdict"]
    record["notes"] = verdict_info["notes"]
    record["n_significant"] = (res["body"] or {}).get("n_significant")
    record["duration_s"] = round(time.time() - t0, 1)
    return record


def write_report(records: List[Dict[str, Any]], ts: str) -> Path:
    report_path = LOG_DIR / f"scenario_test_report_{ts}.md"
    lines = [
        f"# Synthetic Scenario Test Report",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Backend: {BASE}",
        f"",
        f"| # | Scenario | Verdict | n_proteins | n_samples | n_significant | Duration (s) |",
        f"|---|----------|---------|-----------|-----------|---------------|---------------|",
    ]
    for r in records:
        lines.append(
            f"| {r['num']} | {r['name']} | **{r.get('verdict', '?')}** | "
            f"{r.get('n_proteins', '-')} | {r.get('n_samples', '-')} | "
            f"{r.get('n_significant', '-')} | {r.get('duration_s', '-')} |"
        )

    lines.append("")
    lines.append("## Details")
    for r in records:
        lines.append(f"\n### Scenario {r['num']}: {r['name']} — {r.get('verdict', '?')}")
        lines.append(f"- File: `data/synthetic_scenarios/{r['file']}`")
        lines.append(f"- Session: `{r.get('session_id', '-')}`")
        lines.append(f"- Data type detected: {r.get('data_type', '-')}")
        for note in r.get("notes", []):
            lines.append(f"- {note}")
        for cr in r.get("chat_responses", []):
            lines.append(f"\n**Q:** {cr['question']}")
            lines.append(f"- intent: `{cr['intent']}`  status: `{cr['status']}`  http: {cr['status_code']}")
            lines.append(f"- Response excerpt:\n```\n{cr['response_excerpt']}\n```")

    n_pass = sum(1 for r in records if r.get("verdict") == "PASS")
    n_partial = sum(1 for r in records if r.get("verdict") == "PARTIAL")
    n_fail = sum(1 for r in records if r.get("verdict") == "FAIL")
    lines.insert(4, f"**Summary: {n_pass} PASS / {n_partial} PARTIAL / {n_fail} FAIL out of {len(records)}**\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for scenario in SCENARIOS:
        print(f"[{scenario['num']:>2}] Running: {scenario['name']} ...")
        try:
            record = run_scenario(scenario, transcript)
        except Exception as exc:
            record = {"num": scenario["num"], "name": scenario["name"], "file": scenario["file"],
                      "verdict": "FAIL", "notes": [f"exception: {type(exc).__name__}: {exc}"]}
        records.append(record)
        print(f"     -> {record.get('verdict')}: {record.get('notes')}")

    transcript_path = LOG_DIR / f"scenario_test_transcript_{ts}.jsonl"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(json.dumps(entry, default=str) + "\n")

    report_path = write_report(records, ts)
    print(f"\nReport:     {report_path}")
    print(f"Transcript: {transcript_path}")


if __name__ == "__main__":
    main()
