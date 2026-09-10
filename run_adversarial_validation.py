import json
import urllib.request
from pathlib import Path

files = [
    "data/adversarial_tests/1_baseline_normal.csv",
    "data/adversarial_tests/2_trap_anterior_stemi.csv",
    "data/adversarial_tests/3_trap_inferior_stemi.csv",
    "data/adversarial_tests/4_trap_severe_lvh.csv",
]

print("================================================================================")
print("CARDIOAI-12LEAD: IN SILICO ADVERSARIAL PERTURBATION & STRESS-TEST RESULTS")
print("================================================================================")

for f_path in files:
    name = Path(f_path).name
    with open(f_path, "rb") as f:
        content = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8050/api/analyze/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        res = urllib.request.urlopen(req)
        d = json.loads(res.read().decode("utf-8"))
        top = d["top_diagnosis"]
        top_name = d["top_diagnosis_full"]
        prob = round(d["calibrated_probs"][top] * 100, 1)
        raw_prob = round(d["uncalibrated_probs"][top] * 100, 1)
        leads = d["clinical_narrative"]["top_leads"]
        badge = d["triage_badge"]

        print(f"\n[FILE]: {name}")
        print(f"  Classification  : {top_name}")
        print(f"  Calibrated Prob : {prob}% (Raw: {raw_prob}%)")
        print(f"  Triage Status   : {badge}")
        print(f"  Grad-CAM Leads  : {leads}")
        print("  All Probabilities:")
        for k, v in d["calibrated_probs"].items():
            print(f"     - {k:4s}: {round(v*100, 1)}%")

    except Exception as e:
        print(f"Error testing {name}: {e}")

print("\n================================================================================")
