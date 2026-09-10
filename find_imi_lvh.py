import ssl
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sp
import torch
import wfdb

# Set up model
device = torch.device("cpu")
from src.models.resnet1d import build_cardio_resnet1d
from src.explainability.gradcam1d import GradCAM1D

model = build_cardio_resnet1d(num_classes=5, in_channels=12).to(device)
model.load_state_dict(torch.load("models/checkpoints/best_cardio_resnet1d.pth", map_location=device))
model.eval()
gradcam = GradCAM1D(model)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# SSL context
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
req = urllib.request.urlopen(url, context=ctx)
df = pd.read_csv(req, index_col="ecg_id")

def download_file(url_path, dest_path):
    if not dest_path.exists():
        try:
            req = urllib.request.urlopen(url_path, context=ctx)
            dest_path.write_bytes(req.read())
        except Exception as e:
            return False
    return True

def evaluate_record(rec_path_str):
    sub = rec_path_str.split("/")[1]
    name = Path(rec_path_str).name
    dest_dir = Path("data/raw/ptb-xl/records100") / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    url_base = f"https://physionet.org/files/ptb-xl/1.0.3/{rec_path_str}"
    for ext in [".hea", ".dat"]:
        ok = download_file(f"{url_base}{ext}", dest_dir / f"{name}{ext}")
        if not ok:
            return None
    try:
        sig, _ = wfdb.rdsamp(str(dest_dir / name))
    except Exception as e:
        return None

    # Filter
    b, a = sp.butter(2, 0.5 / (100.0 / 2.0), btype="highpass")
    filt_sig = sp.filtfilt(b, a, sig, axis=0).T.astype(np.float32)

    # Inference
    mean = np.mean(filt_sig, axis=1, keepdims=True)
    std = np.std(filt_sig, axis=1, keepdims=True)
    norm = (filt_sig - mean) / (std + 1e-7)
    tensor_in = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor_in)[0]
        cal_probs = torch.sigmoid(logits / 1.28).numpy()
    
    # GradCAM
    cam_mi = gradcam.generate_heatmap(tensor_in, target_class_idx=1)
    cam_hyp = gradcam.generate_heatmap(tensor_in, target_class_idx=4)
    
    def get_attr(cam):
        powers = {lead: float(np.sum(np.abs(filt_sig[i]) * cam)) for i, lead in enumerate(LEAD_NAMES)}
        tot = sum(powers.values()) + 1e-7
        return {l: round((p / tot) * 100, 1) for l, p in powers.items()}
    
    return {
        "cal_probs": cal_probs,
        "mi_attr": get_attr(cam_mi),
        "hyp_attr": get_attr(cam_hyp),
        "sig": filt_sig,
        "name": name,
        "sub": sub
    }

print("--- SCANNING FOR PURE INFERIOR STEMI (IMI) ---", flush=True)
imi_candidates = df[df["scp_codes"].str.contains("'IMI': 100") & ~df["scp_codes"].str.contains("AMI|ASMI|ALMI|CLBBB")]
print(f"Found {len(imi_candidates)} pure IMI cases", flush=True)

best_imi = []
for idx in imi_candidates.index[:25]:
    row = imi_candidates.loc[idx]
    res = evaluate_record(row["filename_lr"])
    if not res:
        continue
    mi_p = float(res["cal_probs"][1])
    attr = res["mi_attr"]
    inf_attr = attr["II"] + attr["III"] + attr["aVF"]
    print(f"Tested IMI {res['name']}: MI={mi_p*100:.1f}%, II+III+aVF={inf_attr:.1f}%", flush=True)
    if mi_p > 0.60:
        best_imi.append({
            "id": idx,
            "name": res["name"],
            "mi_p": mi_p,
            "inf_attr": inf_attr,
            "attr": attr,
            "report": row["report"],
            "age": row["age"],
            "sex": row["sex"]
        })

best_imi.sort(key=lambda x: (x["inf_attr"], x["mi_p"]), reverse=True)
print("\nTOP INFERIOR STEMI CANDIDATES:", flush=True)
for c in best_imi[:5]:
    print(f"ID {c['id']:05d} ({c['name']}) -> MI Prob: {c['mi_p']*100:.1f}% | Inf Leads (II+III+aVF): {c['inf_attr']:.1f}% | Details: II={c['attr']['II']}%, III={c['attr']['III']}%, aVF={c['attr']['aVF']}%", flush=True)
    print(f"   Report: {c['report']}", flush=True)

print("\n--- SCANNING FOR CLASSIC LVH (V5/V6 DOMINANT) ---", flush=True)
lvh_candidates = df[df["scp_codes"].str.contains("'LVH': 100") & ~df["scp_codes"].str.contains("MI|CLBBB|CRBBB")]
print(f"Found {len(lvh_candidates)} pure LVH cases", flush=True)

best_lvh = []
for idx in lvh_candidates.index[:25]:
    row = lvh_candidates.loc[idx]
    res = evaluate_record(row["filename_lr"])
    if not res:
        continue
    hyp_p = float(res["cal_probs"][4])
    attr = res["hyp_attr"]
    v56_attr = attr["V5"] + attr["V6"]
    print(f"Tested LVH {res['name']}: HYP={hyp_p*100:.1f}%, V5+V6={v56_attr:.1f}%", flush=True)
    if hyp_p > 0.70:
        best_lvh.append({
            "id": idx,
            "name": res["name"],
            "hyp_p": hyp_p,
            "v56_attr": v56_attr,
            "attr": attr,
            "report": row["report"],
            "age": row["age"],
            "sex": row["sex"]
        })

best_lvh.sort(key=lambda x: (x["v56_attr"], x["hyp_p"]), reverse=True)
print("\nTOP CLASSIC LVH CANDIDATES:", flush=True)
for c in best_lvh[:5]:
    print(f"ID {c['id']:05d} ({c['name']}) -> HYP Prob: {c['hyp_p']*100:.1f}% | V5+V6: {c['v56_attr']:.1f}% | Details: V4={c['attr']['V4']}%, V5={c['attr']['V5']}%, V6={c['attr']['V6']}%", flush=True)
    print(f"   Report: {c['report']}", flush=True)
