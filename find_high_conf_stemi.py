 import pandas as pd
import wfdb
import torch
import numpy as np
import urllib.request
from pathlib import Path
from src.models.resnet1d import build_cardio_resnet1d

device = torch.device('cpu')
model = build_cardio_resnet1d(num_classes=5, in_channels=12).to(device)
model.load_state_dict(torch.load('models/checkpoints/best_cardio_resnet1d.pth', map_location=device))
model.eval()

df = pd.read_csv('https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv', index_col='ecg_id')
stemi = df[df['scp_codes'].str.contains("'IMI': 100|'AMI': 100") & ~df['scp_codes'].str.contains("CLBBB|CRBBB")]
print(f"Total pure IMI/AMI candidates: {len(stemi)}")

top_candidates = []
for idx in stemi.index[:35]:
    rec_path = stemi.loc[idx, 'filename_lr']
    sub = rec_path.split('/')[1]
    name = Path(rec_path).name
    dest_dir = Path('data/raw/ptb-xl/records100') / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    url_base = f"https://physionet.org/files/ptb-xl/1.0.3/{rec_path}"
    for ext in ['.hea', '.dat']:
        f = dest_dir / f"{name}{ext}"
        if not f.exists():
            try:
                urllib.request.urlretrieve(f"{url_base}{ext}", f)
            except Exception:
                continue
    
    try:
        sig, fields = wfdb.rdsamp(str(dest_dir / name))
    except Exception:
        continue
        
    raw = sig.T.astype(np.float32)
    mean, std = np.mean(raw, axis=1, keepdims=True), np.std(raw, axis=1, keepdims=True)
    norm_sig = (raw - mean) / (std + 1e-7)
    tensor_in = torch.tensor(norm_sig, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cal = torch.sigmoid(model(tensor_in)[0] / 1.28).numpy()
    
    mi_prob = float(cal[1])
    # Check lead voltages for clean morphology
    v2_val = sig[:, fields['sig_name'].index('V2')]
    lead2_val = sig[:, fields['sig_name'].index('II')]
    
    if mi_prob >= 0.90:
        top_candidates.append({
            'id': idx,
            'name': name,
            'path': rec_path,
            'mi_prob': mi_prob,
            'report': stemi.loc[idx, 'report'],
            'scp': stemi.loc[idx, 'scp_codes'],
            'age': stemi.loc[idx, 'age'],
            'sex': stemi.loc[idx, 'sex'],
            'ii_max': float(lead2_val.max()),
            'v2_max': float(v2_val.max()),
        })

print(f"\nFound {len(top_candidates)} candidates with MI calibrated prob >= 90%:")
for c in sorted(top_candidates, key=lambda x: x['mi_prob'], reverse=True):
    print(f"ID {c['id']:05d} ({c['name']}) - MI: {c['mi_prob']*100:.1f}% - Age: {c['age']} Sex: {c['sex']} - II_max: {c['ii_max']:.2f} - V2_max: {c['v2_max']:.2f}")
    print(f"   Report: {c['report']}")
