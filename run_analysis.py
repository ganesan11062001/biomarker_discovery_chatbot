"""
Automation script: create a session, upload data, run 7 analysis steps.
Usage: python run_analysis.py
"""
import json
import sys
import time
import requests

API = "http://localhost:8000"
DATA_FILE = "data/raw/dac6b726-ea21-4f7e-a372-978d547ef527/a445fefdc85c4480a7f5d0e3ae476790.xlsx"

MESSAGES = [
    "Compare WT and mdx group",
    "List the top 10 biomarkers",
    "Compare uDys5 and mdx group, identify top 10 biomarkers, then run pathway analysis",
    "Compare the list of differential biomarkers between WT and mdx groups and between uDys5 and mdx groups. Are there any overlap?",
]


def step(label: str, n: int, total: int):
    print(f"\n{'='*70}")
    print(f"[{n}/{total}] {label}")
    print('='*70)


def main():
    # 1. Create session
    r = requests.post(f"{API}/chat/session")
    r.raise_for_status()
    session_id = r.json()["session_id"]
    print(f"Session created: {session_id}")

    # 2. Upload data file
    print(f"Uploading data file: {DATA_FILE}")
    with open(DATA_FILE, "rb") as fh:
        r = requests.post(
            f"{API}/upload/",
            files={"file": ("data.xlsx", fh, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"session_id": session_id,
                  "disease_program": "Duchenne Muscular Dystrophy (DMD)",
                  "organism": "Mus musculus"},
        )
    if not r.ok:
        print(f"Upload failed: {r.status_code} {r.text}")
        sys.exit(1)
    upload_info = r.json()
    print(f"Upload OK — {upload_info['n_proteins']} proteins, {upload_info['n_samples']} samples")
    print(f"Groups detected: {upload_info.get('inferred_groups') or upload_info.get('label_map')}")

    # 3. Send each message
    results = []
    for i, msg in enumerate(MESSAGES, 1):
        step(msg, i, len(MESSAGES))
        payload = {"session_id": session_id, "message": msg}
        t0 = time.time()
        r = requests.post(f"{API}/chat/", json=payload, timeout=300)
        elapsed = time.time() - t0
        if not r.ok:
            print(f"ERROR {r.status_code}: {r.text[:500]}")
            results.append({"step": i, "message": msg, "error": r.text[:200]})
            continue
        data = r.json()
        response = data.get("response", "")
        intent = data.get("intent", "")
        status_ = data.get("status", "")
        print(f"Intent: {intent}  |  Status: {status_}  |  Time: {elapsed:.1f}s")
        print(f"\nResponse:\n{response[:2000]}")
        if len(response) > 2000:
            print(f"... [{len(response)-2000} more chars]")
        results.append({
            "step": i, "message": msg, "intent": intent,
            "status": status_, "response": response, "elapsed": elapsed,
        })

    # 4. Save results
    out_file = f"outputs/analysis_run_{session_id[:8]}.json"
    import os; os.makedirs("outputs", exist_ok=True)
    with open(out_file, "w") as fh:
        json.dump({"session_id": session_id, "results": results}, fh, indent=2, default=str)
    print(f"\n\nAll done. Results saved to {out_file}")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()
