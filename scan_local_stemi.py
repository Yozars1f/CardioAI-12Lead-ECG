import wfdb
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.resnet1d import build_cardio_resnet1d

device = torch.device('cpu')
model = build_cardio_resnet1d(num_classes=5, in_channels=12).to(device)
model.load_state_dict(torch.load('models/checkpoints/best_cardio_resnet1d.pth', map_location=device))
model.eval()

df = pd.read_csv('data/raw/ptb-xl/ptbxl_database.csv', index_col='ecg_id')

all_local = list(Path('data/raw/ptb-xl/records100').rglob('*.dat'))
print(f"Total local records: {len(all_local)}")

high_mi = []
for p in all_local:
    rec_path = p.with_suffix('')
    try:
        sig, fields = wfdb.rdsamp(str(rec_path))
    except Exception:
        continue
    
    raw = sig.T.astype(np.float32)
    mean, std = np.mean(raw, axis=1, keepdims=True), np.std(raw, axis=1, keepdims=True)
    norm_sig = (raw - mean) / (std + 1e-7)
    tensor_in = torch.tensor(norm_sig, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cal = torch.sigmoid(model(tensor_in)[0] / 1.28).numpy()
    
    mi_p = float(cal[1])
    if mi_p >= 0.88:
        rec_name = p.stem
        try:
            eid = int(rec_name.split('_')[0])
            scp = str(df.loc[eid, 'scp_codes']) if eid in df.index else 'N/A'
            rep = str(df.loc[eid, 'report']) if eid in df.index else 'N/A'
            age = df.loc[eid, 'age'] if eid in df.index else 'N/A'
            sex = df.loc[eid, 'sex'] if eid in df.index else 'N/A'
        except Exception:
            scp, rep, age, sex = 'N/A', 'N/A', 'N/A', 'N/A'
            
        high_mi.append({
            'name': rec_name,
            'mi_prob': mi_p,
            'scp': scp,
            'report': rep,
            'age': age,
            'sex': sex,
            'sig': sig
        })

print(f"\nFound {len(high_mi)} local records with MI Calibrated Prob >= 88%:")
for c in sorted(high_mi, key=lambda x: x['mi_prob'], reverse=True):
    print(f"{c['name']} -- MI: {c['mi_prob']*100:.1f}% -- Age: {c['age']} Sex: {c['sex']} -- SCP: {c['scp']}")
    print(f"   Report: {c['report']}")
