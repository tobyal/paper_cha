"""Collect current tri-plane backend receipts; old MLP tests stay historical."""
import hashlib
import json
import sys
from pathlib import Path
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.support import SurfaceSupportLearner
from ssr.geometry import write_json, load_points

prior = ROOT / 'outputs/smoke_prior/last.pt'
learner = SurfaceSupportLearner(checkpoint=prior)
assert learner.provenance['state_items'] == 481
fixed = ROOT / 'outputs/triplane_modes/frozen'
assert (load_points(fixed / 'support_initial.ply') == load_points(fixed / 'support_final.ply')).all()
receipts = {}
for run in ['triplane_modes/joint', 'triplane_modes/frozen', 'triplane_modes/raw', 'triplane_modes/external']:
    receipt = json.loads((ROOT / 'outputs' / run / 'status.json').read_text())
    assert receipt['status'] == 'complete', receipt
    receipts[run] = receipt
source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in ROOT.rglob('*.py') if 'vendor' not in path.parts}
result = {'status': 'passed', 'purpose': 'Implementation validation only; no reconstruction-quality claim',
          'backend_version': 2,
          'contracts': json.loads((ROOT / 'verification_triplane.json').read_text()),
          'pretrained_from_new_prior_strict_load': True, 'actual_frozen_support_unchanged': True,
          'runs': receipts, 'multi_gpu': json.loads((ROOT / 'outputs/triplane_modes/status.json').read_text()),
          'source_hashes': source_hashes}
write_json(ROOT / 'VALIDATION_TRIPLANE.json', result)
with zipfile.ZipFile(ROOT / 'validation_triplane_artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['verification_triplane.json', 'VALIDATION_TRIPLANE.json']:
        archive.write(ROOT / name, name)
    for path in (ROOT / 'outputs/triplane_modes').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.ply', '.npz', '.log'):
            archive.write(path, str(path.relative_to(ROOT)))
print(json.dumps({'status': 'passed', 'runs': len(receipts), 'archive': str(ROOT / 'validation_triplane_artifacts.zip')}))
