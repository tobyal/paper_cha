"""Check full-prior checkpoint round trips, including trained direct heads."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import patch_layout, write_json
from ssr.support import SurfaceSupportLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', help='Optional staged ssr/support.py for isolated validation')
    args = parser.parse_args()
    learner_type = SurfaceSupportLearner
    if args.candidate:
        spec = importlib.util.spec_from_file_location('ssr._candidate_support', args.candidate)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        candidate.ROOT = ROOT
        learner_type = candidate.SurfaceSupportLearner
    torch.set_num_threads(2)
    torch.manual_seed(21)
    points = torch.randn(100, 3, device='cuda') * .2
    layout = patch_layout(points, patch_size=64, overlap=1)
    result = {}
    for decoder in ('query', 'direct'):
        model = learner_type(decoder=decoder).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        prediction = model(layout)
        (prediction['points'].square().mean() + .001 * prediction['regularizer']).backward()
        optimizer.step()
        with torch.no_grad():
            expected = model(layout)['points']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prior.pt'
            torch.save({'support_network': model.state_dict(), 'backbone': model.backbone.state_dict(),
                        'support_config': {'decoder': decoder, 'deform': True},
                        'optimizer': optimizer.state_dict()}, path)
            loaded = learner_type(checkpoint=path, decoder=decoder).cuda()
            with torch.no_grad():
                actual = loaded(layout)['points']
            assert torch.equal(expected, actual)
            assert loaded.provenance['complete_prior_loaded']
            assert not loaded.provenance['new_random_head']
            try:
                learner_type(checkpoint=path, decoder=decoder, deform=False)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched support configuration was accepted')
            result[decoder] = {'output_max_error': float((expected - actual).abs().max()),
                               'state_items': len(loaded.state_dict()), 'complete_prior_loaded': True}
    result['status'] = 'passed'
    write_json(ROOT / 'verification_complete_prior.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
